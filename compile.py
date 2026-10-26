"""A small compiler for Emmy pointwise kernels and f32 array/row reductions."""

import json
import math
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import (
    Accum, Assign, BinaryExpr, Cond, KernelOp, Literal, Load, Smem,
    StridedLoop, Sync, Tile, TreeHalve, Var, WarpShuffle, Write,
)


# IR dtype -> register group, PTX arithmetic type, bytes per array element.
TYPES = {"f16": ("h", "b16", 2), "f32": ("f", "f32", 4),
         "f64": ("d", "f64", 8), "i32": ("r", "s32", 4),
         "i64": ("rd", "s64", 8)}


class Compiler:
    # Prepare the output list, register counters, and name-to-register mappings.
    def __init__(self):
        self.lines = []
        # p: conditions; r: 32-bit integers; rd: 64-bit integers/addresses; f: f32 values.
        self.counts = {"p": 0, "r": 0, "rd": 0, "f": 0}
        self.counts.update({"h": 0, "d": 0})  # Half-storage bits and double values.
        self.values = {}       # IR value name -> register name
        self.value_types = {}  # IR value name -> dtype name
        self.buffers = {}      # Buffer name -> register holding its address
        self.buffer_types = {} # Buffer name -> array element dtype
        self.buffer_shapes = {} # Static shapes for row-major multidimensional accesses
        self.shared = {}       # Shared buffer name -> declaration symbol
        self.shared_declarations = []
        self.carried = {}      # Accumulator name -> stable register across iterations
        self.label_count = 0
        self.block_threads = None
        self.control_depth = 0 # Collectives must be outside divergent loops/conditions.

    # Visit nested statements when collecting buffer declarations and types.
    def walk(self, body):
        for stmt in body:
            yield stmt
            if isinstance(stmt, StridedLoop):
                yield from self.walk(stmt.body)
            elif isinstance(stmt, Cond):
                yield from self.walk(stmt.body)
                yield from self.walk(stmt.else_body)

    # Give each generated branch target a distinct name.
    def label(self, prefix):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    # Record a result, retaining the stable register of a loop accumulator.
    def bind(self, name, value, dtype):
        if name in self.carried:
            if dtype != "f32":
                raise NotImplementedError("Reduction state must remain f32")
            target = self.carried[name]
            if target != value:
                self.emit(f"mov.f32 {target}, {value};")
            value = target
        self.values[name], self.value_types[name] = value, dtype

    # Append one PTX instruction to the output list
    # Super simple helper function
    def emit(self, instruction):
        self.lines.append("    " + instruction)

    # Return a fresh register name, such as %r0, and update its counter
    # Helper funciton to makes things easier
    def register(self, kind):
        name = f"%{kind}{self.counts[kind]}"
        self.counts[kind] += 1
        return name

    # Translate an index calculation and return its register name or constant.
    def expression(self, expr):
        # case 1: the expression is a variable
        if isinstance(expr, Var):
            dtype = self.value_types[expr.name]
            if dtype not in ("i32", "i64"):
                raise NotImplementedError(f"Index variable {expr.name} is not an integer")
            return self.convert(self.values[expr.name], dtype, "i64")
        # case 2, the variable is an int, ie directly a value
        if isinstance(expr, Literal) and expr.dtype == "int":
            if not -(2**63) <= expr.value < 2**63:
                raise NotImplementedError("Index literal does not fit signed 64 bits")
            return str(expr.value)
        # case 3: computation has to be done to get the expression
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*", "/", "//", "%"):
            left = self.expression(expr.left)
            right = self.expression(expr.right)
            if expr.op in ("/", "//", "%"):
                if isinstance(expr.right, Literal) and expr.right.value == 0:
                    raise ValueError("Index division by zero")
                quotient, remainder = self.floor_divmod(left, right)
                return remainder if expr.op == "%" else quotient
            result = self.register("rd")
            operation = {"+": "add.s64", "-": "sub.s64", "*": "mul.lo.s64"}[expr.op]
            self.emit(f"{operation} {result}, {left}, {right};")
            return result
        raise NotImplementedError(f"Unsupported index expression: {expr}")

    # Correct PTX's truncating division to Emmy's floor division and remainder.
    def floor_divmod(self, left, right):
        quotient, remainder, signs = [self.register("rd") for _ in range(3)]
        nonzero, different, adjust = [self.register("p") for _ in range(3)]
        self.emit(f"div.s64 {quotient}, {left}, {right};")
        self.emit(f"rem.s64 {remainder}, {left}, {right};")
        self.emit(f"setp.ne.s64 {nonzero}, {remainder}, 0;")
        self.emit(f"xor.b64 {signs}, {remainder}, {right};")
        self.emit(f"setp.lt.s64 {different}, {signs}, 0;")
        self.emit(f"and.pred {adjust}, {nonzero}, {different};")
        self.emit(f"@{adjust} sub.s64 {quotient}, {quotient}, 1;")
        self.emit(f"@{adjust} add.s64 {remainder}, {remainder}, {right};")
        return quotient, remainder

    # Generate PTX to find an array element's address and return its register name.
    def address(self, buffer, indices):
        if len(indices) > 1:
            shape = self.buffer_shapes.get(buffer)
            if shape is None or len(shape) != len(indices):
                raise NotImplementedError(f"Multidimensional access needs a matching shape for {buffer}")
            flat = indices[0]
            for extent, index in zip(shape[1:], indices[1:]):
                flat = BinaryExpr("+", BinaryExpr("*", flat, Literal(extent, "int")), index)
            indices = (flat,)
        
        # we do the necessary computation on the index ("(a0 * 2) + 0")
        index = self.expression(indices[0]) if indices else "0"

        offset = self.register("rd")
        address = self.register("rd")
        nbytes = TYPES[self.buffer_types[buffer]][2]
        self.emit(f"mul.lo.s64 {offset}, {index}, {nbytes};")
        self.emit(f"add.u64 {address}, {self.buffers[buffer]}, {offset};")
        return address

    # Convert one register's numeric value, preserving its declared source type.
    def convert(self, value, source, destination):
        if source not in TYPES or destination not in TYPES:
            raise NotImplementedError(f"Unsupported conversion: {source} -> {destination}")
        if source == destination:
            return value
        result = self.register(TYPES[destination][0])
        rounding = ""
        if destination.startswith("i") and source.startswith("f"):
            rounding = ".rzi"  # Float-to-integer conversion truncates toward zero.
        elif destination.startswith("f") and (source.startswith("i") or TYPES[destination][2] < TYPES[source][2]):
            rounding = ".rn"
        target = "f16" if destination == "f16" else TYPES[destination][1]
        origin = "f16" if source == "f16" else TYPES[source][1]
        self.emit(f"cvt{rounding}.{target}.{origin} {result}, {value};")
        return result

    # Emit one scalar operation; f16 results round back after each IR Assign.
    def arithmetic(self, operation, args, dtype):
        work_type = "f32" if dtype == "f16" else dtype
        suffix = TYPES[work_type][1]
        floating = work_type.startswith("f")
        result = self.register(TYPES[work_type][0])
        if operation in ("add", "subtract", "multiply", "divide"):
            instruction = {"add": "add", "subtract": "sub", "multiply": "mul", "divide": "div"}[operation]
            instruction += ".rn" if floating else ".lo" if operation == "multiply" else ""
            self.emit(f"{instruction}.{suffix} {result}, {args[0]}, {args[1]};")
        elif operation in ("minimum", "maximum"):
            comparison = self.register("p")
            test = "lt" if operation == "minimum" else "gt"
            if dtype == "f16":
                test = "le" if operation == "minimum" else "ge"  # Half ties keep the first input.
            self.emit(f"setp.{test}.{suffix} {comparison}, {args[0]}, {args[1]};")
            self.emit(f"selp.{suffix} {result}, {args[0]}, {args[1]}, {comparison};")
            if floating:
                # Propagate a NaN on either side, including when the first comparison fails.
                unordered = self.register("p")
                self.emit(f"setp.nan.{suffix} {unordered}, {args[0]}, {args[0]};")
                self.emit(f"@{unordered} mov.{suffix} {result}, {args[0]};")
        elif operation == "relu":
            zero = "0d0000000000000000" if work_type == "f64" else "0f00000000" if floating else "0"
            negative = self.register("p")
            test = "le" if dtype == "f16" else "lt"
            self.emit(f"setp.{test}.{suffix} {negative}, {args[0]}, {zero};")
            self.emit(f"selp.{suffix} {result}, {zero}, {args[0]}, {negative};")
        elif operation == "sqrt" and floating:
            self.emit(f"sqrt.rn.{suffix} {result}, {args[0]};")
        elif operation in ("exp", "exp_fast") and work_type == "f32":
            scaled = self.register("f")
            self.emit(f"mul.rn.f32 {scaled}, {args[0]}, 0f3FB8AA3B;")
            self.emit(f"ex2.approx.f32 {result}, {scaled};")
        elif operation == "copy":
            self.emit(f"mov.{suffix} {result}, {args[0]};")
        else:
            raise NotImplementedError(f"Unsupported arithmetic: {operation} for {dtype}")
        return self.convert(result, work_type, dtype)

    # Compile an integer comparison or logical expression into a predicate register.
    def predicate(self, expr):
        if not isinstance(expr, BinaryExpr):
            raise NotImplementedError(f"Unsupported condition: {expr}")
        result = self.register("p")
        comparisons = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
        if expr.op in comparisons:
            left, right = self.expression(expr.left), self.expression(expr.right)
            self.emit(f"setp.{comparisons[expr.op]}.s64 {result}, {left}, {right};")
        elif expr.op in ("and", "or", "&&", "||"):
            left, right = self.predicate(expr.left), self.predicate(expr.right)
            operation = "and" if expr.op in ("and", "&&") else "or"
            self.emit(f"{operation}.pred {result}, {left}, {right};")
        else:
            raise NotImplementedError(f"Unsupported condition operator: {expr.op}")
        return result

    # Compile both conditional bodies without leaking their local value names.
    def conditional(self, stmt):
        predicate = self.predicate(stmt.cond)
        otherwise, done = self.label("ELSE"), self.label("ENDIF")
        values, types = self.values.copy(), self.value_types.copy()
        self.emit(f"@!{predicate} bra {otherwise};")
        self.control_depth += 1
        for child in stmt.body:
            self.statement(child)
        self.emit(f"bra {done};")
        self.lines.append(otherwise + ":")
        self.values, self.value_types = values.copy(), types.copy()
        for child in stmt.else_body:
            self.statement(child)
        self.lines.append(done + ":")
        self.control_depth -= 1
        self.values, self.value_types = values, types

    # Seed f32 running sums and translate one strided loop with a stable counter.
    def strided_loop(self, stmt):
        if not isinstance(stmt.step, Literal) or stmt.step.value <= 0:
            raise NotImplementedError("StridedLoop needs a positive constant step")
        # Inner loops own their own seeds; only descend through conditions here.
        def accumulators(body):
            for child in body:
                if isinstance(child, Accum):
                    yield child
                elif isinstance(child, Cond):
                    yield from accumulators(child.body)
                    yield from accumulators(child.else_body)
        seen = set()
        for acc in accumulators(stmt.body):
            if acc.op.name != "add" or acc.dtype is None or acc.dtype.name != "f32" or acc.base is not None:
                raise NotImplementedError("Only ordinary f32 additive accumulators are supported")
            if acc.name in seen:
                continue
            seen.add(acc.name)
            if stmt.seed:
                target = self.carried.setdefault(acc.name, self.register("f"))
                self.emit(f"mov.f32 {target}, 0f00000000;")
                self.bind(acc.name, target, "f32")
            elif acc.name not in self.carried:
                raise NotImplementedError("Unseeded loop requires an existing accumulator")
        start = self.expression(stmt.start)
        step = self.expression(stmt.step)
        bound = self.expression(stmt.end) if stmt.end is not None else str(stmt.axis.extent.value)
        counter, predicate = self.register("rd"), self.register("p")
        begin, done = self.label("LOOP"), self.label("ENDLOOP")
        values, types = self.values.copy(), self.value_types.copy()
        self.emit(f"mov.s64 {counter}, {start};")
        self.values[stmt.axis.name], self.value_types[stmt.axis.name] = counter, "i64"
        self.lines.append(begin + ":")
        self.emit(f"setp.ge.s64 {predicate}, {counter}, {bound};")
        self.emit(f"@{predicate} bra {done};")
        self.control_depth += 1
        for child in stmt.body:
            self.statement(child)
        self.control_depth -= 1
        self.emit(f"add.s64 {counter}, {counter}, {step};")
        self.emit(f"bra {begin};")
        self.lines.append(done + ":")
        self.values, self.value_types = values, types

    # Check the scalar sum combine used by the supported reduction schedules.
    def check_combine(self, stmt):
        if self.control_depth or self.block_threads is None:
            raise NotImplementedError("Reduction collectives require an unconditional cooperative Tile")
        if stmt.dtype.name != "f32" or len(stmt.state) != 1 or len(stmt.state_b) != 1:
            raise NotImplementedError("Only scalar f32 reduction states are supported")
        if stmt.length < 1 or stmt.length & (stmt.length - 1):
            raise NotImplementedError("Reduction combine length must be a power of two")
        if len(stmt.combine_states) != 1:
            raise NotImplementedError("Only additive scalar combines are supported")
        merge = stmt.combine_states[0]
        if (not isinstance(merge, Assign) or merge.name != stmt.state[0]
                or merge.op.name != "add" or tuple(merge.args) != (stmt.state[0], stmt.state_b[0])
                or (merge.dtype is not None and merge.dtype.name != "f32")
                or self.value_types.get(stmt.state[0]) != "f32"):
            raise NotImplementedError("Only f32 state-plus-neighbor combines are supported")

    # Combine each aligned group of lanes using the IR's XOR shuffle ladder.
    def warp_shuffle(self, stmt):
        self.check_combine(stmt)
        if stmt.length > 32:
            raise NotImplementedError("WarpShuffle length cannot exceed 32")
        mask = self.register("r")
        self.emit(f"activemask.b32 {mask};")
        state, neighbor = stmt.state[0], stmt.state_b[0]
        stride = stmt.length // 2
        while stride:
            bits, shuffled, value = self.register("r"), self.register("r"), self.register("f")
            self.emit(f"mov.b32 {bits}, {self.values[state]};")
            self.emit(f"shfl.sync.bfly.b32 {shuffled}, {bits}, {stride}, 31, {mask};")
            self.emit(f"mov.b32 {value}, {shuffled};")
            self.bind(neighbor, value, "f32")
            self.statement(stmt.combine_states[0], "f32")
            stride //= 2

    # Reduce the shared per-warp partials, then load the root in every thread.
    def tree_halve(self, stmt):
        self.check_combine(stmt)
        if (stmt.inner is not None or stmt.barrier_id != 0 or stmt.barrier_count is not None
                or stmt.tid_var != "warp" or len(stmt.bufs) != 1
                or stmt.length > self.block_threads // 32):
            raise NotImplementedError("Only a flat per-warp tree with block-wide barriers is supported")
        buffer, state, neighbor = stmt.bufs[0], stmt.state[0], stmt.state_b[0]
        if buffer not in self.shared or self.buffer_types[buffer] != "f32":
            raise NotImplementedError("TreeHalve requires an f32 shared buffer")
        if self.buffer_shapes[buffer] != (stmt.length,):
            raise NotImplementedError("TreeHalve requires one shared slot per partial state")
        stride = stmt.length // 2
        while stride:
            skip = self.label("TREE_SKIP")
            leader, active, participates = [self.register("p") for _ in range(3)]
            self.emit(f"setp.eq.s64 {leader}, {self.values['lane']}, 0;")
            self.emit(f"setp.lt.s64 {active}, {self.values[stmt.tid_var]}, {stride};")
            self.emit(f"and.pred {participates}, {leader}, {active};")
            self.emit(f"@!{participates} bra {skip};")
            left = self.address(buffer, (Var(stmt.tid_var),))
            right = self.address(buffer, (BinaryExpr("+", Var(stmt.tid_var), Literal(stride, "int")),))
            other = self.register("f")
            self.emit(f"ld.shared.f32 {self.values[state]}, [{left}];")
            self.emit(f"ld.shared.f32 {other}, [{right}];")
            self.bind(neighbor, other, "f32")
            self.statement(stmt.combine_states[0], "f32")
            self.emit(f"st.shared.f32 [{left}], {self.values[state]};")
            self.lines.append(skip + ":")
            self.emit("bar.sync 0;")
            stride //= 2
        root = self.address(buffer, (Literal(0, "int"),))
        self.emit(f"ld.shared.f32 {self.values[state]}, [{root}];")

    # Generate PTX for a typed computation, nested control flow, or collective.
    def statement(self, stmt, default_dtype=None):
        if isinstance(stmt, Load):
            dtype = self.buffer_types[stmt.input]
            if stmt.dtype is not None and stmt.dtype.name != dtype:
                raise ValueError(f"Load dtype disagrees with buffer {stmt.input}")
            kind, suffix, nbytes = TYPES[dtype]
            address = self.address(stmt.input, stmt.index)
            for i, name in enumerate(stmt.names):
                result = self.register(kind)
                space = "shared" if stmt.input in self.shared else "global"
                self.emit(f"ld.{space}.{suffix} {result}, [{address}+{nbytes * i}];")
                self.bind(name, result, dtype)
        elif isinstance(stmt, Assign):
            dtype = stmt.dtype.name if stmt.dtype is not None else default_dtype
            if dtype not in TYPES:
                raise NotImplementedError(f"Unsupported Assign dtype: {stmt.dtype}")
            if stmt.op.name == "copy":
                # Cast directly: an intermediate f32 can change f64-to-f16 rounding.
                name = stmt.args[0]
                result = self.convert(self.values[name], self.value_types[name], dtype)
            else:
                work_type = "f32" if dtype == "f16" else dtype
                typed_args = [self.convert(self.values[name], self.value_types[name], dtype) for name in stmt.args]
                args = [self.convert(value, dtype, work_type) for value in typed_args]
                result = self.arithmetic(stmt.op.name, args, dtype)
            self.bind(stmt.name, result, dtype)
        elif isinstance(stmt, Write) and not stmt.atomic and stmt.swizzle == "NONE":
            dtype = self.buffer_types[stmt.output]
            _, suffix, nbytes = TYPES[dtype]
            address = self.address(stmt.output, stmt.index)
            for i, name in enumerate(stmt.values):
                value = self.convert(self.values[name], self.value_types[name], dtype)
                space = "shared" if stmt.output in self.shared else "global"
                self.emit(f"st.{space}.{suffix} [{address}+{nbytes * i}], {value};")
        elif isinstance(stmt, Accum):
            if (stmt.name not in self.carried or stmt.op.name != "add" or stmt.base is not None
                    or stmt.dtype is None or stmt.dtype.name != "f32"):
                raise NotImplementedError("Accum requires a seeded f32 running sum")
            value = self.convert(self.values[stmt.value], self.value_types[stmt.value], "f32")
            target = self.carried[stmt.name]
            self.emit(f"add.rn.f32 {target}, {target}, {value};")
        elif isinstance(stmt, StridedLoop):
            self.strided_loop(stmt)
        elif isinstance(stmt, Cond):
            self.conditional(stmt)
        elif isinstance(stmt, Smem):
            pass  # Its declaration and base address are emitted before the body.
        elif isinstance(stmt, Sync):
            if (self.block_threads is None or self.control_depth or stmt.warp
                    or stmt.barrier_id != 0 or stmt.count is not None):
                raise NotImplementedError("Only unconditional block-wide Sync is supported")
            self.emit("bar.sync 0;")
        elif isinstance(stmt, WarpShuffle):
            self.warp_shuffle(stmt)
        elif isinstance(stmt, TreeHalve):
            self.tree_halve(stmt)
        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    # Build the complete PTX function: arguments, thread mapping, body, and declarations.
    def kernel(self, kernel, argument_order, buffer_dtypes=None, buffer_shapes=None):
        if kernel.serial or len(kernel.body) != 1:
            raise NotImplementedError("Expected one simple pointwise Tile")
        tile = kernel.body[0]
        if (not isinstance(tile, Tile) or not tile.axes or tile.aux_threads
                or tile.raster_axes is not None):
            raise NotImplementedError("Expected one static Tile without auxiliary threads or raster scheduling")

        self.block_threads = tile.block_threads
        extents = [axis.extent.value for axis in tile.axes]
        if any(not isinstance(e, int) or e < 0 for e in extents) or (len(extents) > 1 and 0 in extents):
            raise NotImplementedError("Tile axes must be static; empty Tiles are supported only for one flat axis")
        extent = math.prod(extents)
        if extent >= 2**63:
            raise NotImplementedError("Tile extent must be a static signed-64-bit size")
        if self.block_threads is not None:
            if (not isinstance(self.block_threads, int) or not 32 <= self.block_threads <= 1024
                    or self.block_threads % 32 or extent == 0 or extent % self.block_threads
                    or extents[-1] != self.block_threads):
                raise NotImplementedError("Reduction Tiles need complete blocks and one innermost cooperative axis")

        self.buffer_types = dict(buffer_dtypes or {})
        self.buffer_shapes = {name: tuple(shape) for name, shape in (buffer_shapes or {}).items()}
        if any(any(not isinstance(e, int) or e < 0 for e in shape) for shape in self.buffer_shapes.values()):
            raise NotImplementedError("Buffer shapes must have nonnegative static extents")
        for stmt in self.walk(tile.body):
            if isinstance(stmt, Load) and stmt.dtype is not None:
                self.buffer_types.setdefault(stmt.input, stmt.dtype.name)
            elif isinstance(stmt, Write) and stmt.value_dtype is not None:
                self.buffer_types.setdefault(stmt.output, stmt.value_dtype.name)
        for name in argument_order:
            if self.buffer_types.get(name) not in TYPES:
                raise NotImplementedError(f"Unsupported or missing dtype for buffer {name}")
        # Read the array addresses passed by the launcher
        # (the 3 input adresses/paramter in p00)
        for i, name in enumerate(argument_order):
            address = self.register("rd")
            self.emit(f"ld.param.u64 {address}, [arg_{i}];")
            self.emit(f"cvta.to.global.u64 {address}, {address};")
            self.buffers[name] = address

        for stmt in self.walk(tile.body):
            if not isinstance(stmt, Smem):
                continue
            if (stmt.name in self.shared or stmt.name in argument_order or stmt.dtype != "float"
                    or self.buffer_types.get(stmt.name, "f32") != "f32" or any(e <= 0 for e in stmt.extents)):
                raise NotImplementedError("Shared buffers must be distinct, positive-sized float arrays")
            symbol = f"shared_{len(self.shared)}"
            align = stmt.align or 4
            if align < 4 or align & (align - 1):
                raise NotImplementedError("Shared alignment must be a power of two, at least four")
            self.shared[stmt.name] = symbol
            self.buffer_types[stmt.name] = "f32"
            self.buffer_shapes[stmt.name] = tuple(stmt.extents)
            self.shared_declarations.append(f".shared .align {align} .b8 {symbol}[{4 * math.prod(stmt.extents)}];")
            base, address = self.register("r"), self.register("rd")
            self.emit(f"mov.u32 {base}, {symbol};")
            self.emit(f"cvt.u64.u32 {address}, {base};")
            self.buffers[stmt.name] = address

        # Tile: calculate this thread's coordinate and skip excess threads.
        block, size, thread = [self.register("r") for _ in range(3)]
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {size}, %ntid.x;")
        self.emit(f"mov.u32 {thread}, %tid.x;")
        if self.block_threads is not None:
            lane, warp = self.register("r"), self.register("r")
            self.emit(f"and.b32 {lane}, {thread}, 31;")
            self.emit(f"shr.u32 {warp}, {thread}, 5;")
            self.values.update(lane=lane, warp=warp)
            self.value_types.update(lane="i32", warp="i32")
            self.values["lane"] = self.convert(lane, "i32", "i64")
            self.values["warp"] = self.convert(warp, "i32", "i64")
            self.value_types.update(lane="i64", warp="i64")
        wide_block, wide_size, wide_thread, coordinate = [self.register("rd") for _ in range(4)]
        self.emit(f"cvt.u64.u32 {wide_block}, {block};")
        self.emit(f"cvt.u64.u32 {wide_size}, {size};")
        self.emit(f"cvt.u64.u32 {wide_thread}, {thread};")
        self.emit(f"mul.lo.s64 {coordinate}, {wide_block}, {wide_size};")
        self.emit(f"add.s64 {coordinate}, {coordinate}, {wide_thread};")
        predicate = self.register("p")
        self.emit(f"setp.ge.u64 {predicate}, {coordinate}, {extent};")
        self.emit(f"@{predicate} bra DONE;")
        # Decode the linear cell into row-major Tile coordinates.
        for i, axis in enumerate(tile.axes):
            stride = math.prod(extents[i + 1:])
            value = coordinate
            if stride != 1:
                value = self.register("rd")
                self.emit(f"div.u64 {value}, {coordinate}, {stride};")
            if i:
                remainder = self.register("rd")
                self.emit(f"rem.u64 {remainder}, {value}, {extents[i]};")
                value = remainder
            self.values[axis.name], self.value_types[axis.name] = value, "i64"

        # Translate the IR KernelOp's tile to ptx.
        for stmt in tile.body:
            self.statement(stmt)
        self.lines.append("DONE:")
        self.emit("ret;")

        # We now know how many register names the generated instructions use.
        types = {"p": "pred", "r": "b32", "rd": "b64", "f": "f32", "h": "b16", "d": "f64"}
        declarations = [f"    .reg .{types[kind]} %{kind}<{count}>;"
                        for kind, count in self.counts.items() if count]
        parameters = ",\n".join(f"    .param .u64 arg_{i}"
                                  for i in range(len(argument_order)))
        return "\n".join([
            ".version 8.0", ".target sm_89", ".address_size 64", "",
            *self.shared_declarations,
            f".visible .entry {kernel.name}(\n{parameters}\n)", "{",
            *declarations, "", *self.lines, "}", "",
        ])


def main():
    if len(sys.argv) != 2:
        sys.exit("Two arguments are required, to be used as follows: " \
        "python compile.py path/to/kernel.ir.json")
    ir_path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
    if len(kernels) != 1:
        raise NotImplementedError("Expected exactly one kernel in the IR file")
    kernel = kernels[0]
    metadata = json.loads(ir_path.with_name("meta.json").read_text(encoding="utf-8"))
    launch = next(item for item in metadata["kernels"] if item["entry"] == kernel.name)
    tile = kernel.body[0]
    if isinstance(tile, Tile) and tile.block_threads is not None:
        if launch["block"] != [tile.block_threads, 1, 1] or launch["grid"][1:] != [1, 1]:
            raise NotImplementedError("Cooperative launch must match block_threads and use x dimensions only")
        if launch["grid"][0] * tile.block_threads < math.prod(axis.extent.value for axis in tile.axes):
            raise ValueError("Launch does not cover the cooperative Tile")
    if any(arg.get("scalar") for arg in launch["args"]):
        raise NotImplementedError("Only buffer arguments are supported")
    buffer_dtypes = {arg["name"]: arg["dtype"] for arg in launch["args"]}
    buffer_shapes = {arg["name"]: arg["shape"] for arg in launch["args"]}
    print(Compiler().kernel(kernel, launch["arg_order"], buffer_dtypes, buffer_shapes), end="")


if __name__ == "__main__":
    main()
