"""A small compiler for flat, statically sized Emmy pointwise kernels."""

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import (
    Assign, BinaryExpr, KernelOp, Literal, Load, Tile, Var, Write,
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
        if len(indices) != 1:
            raise NotImplementedError("Only one-dimensional indices are supported")
        
        # we do the necessary computation on the index ("(a0 * 2) + 0")
        index = self.expression(indices[0])

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

    # Generate PTX for one typed Load, Assign, or Write from the Tile body.
    def statement(self, stmt):
        if isinstance(stmt, Load):
            dtype = self.buffer_types[stmt.input]
            if stmt.dtype is not None and stmt.dtype.name != dtype:
                raise ValueError(f"Load dtype disagrees with buffer {stmt.input}")
            kind, suffix, nbytes = TYPES[dtype]
            address = self.address(stmt.input, stmt.index)
            for i, name in enumerate(stmt.names):
                result = self.register(kind)
                self.emit(f"ld.global.{suffix} {result}, [{address}+{nbytes * i}];")
                self.values[name], self.value_types[name] = result, dtype
        elif isinstance(stmt, Assign):
            if stmt.dtype is None or stmt.dtype.name not in TYPES:
                raise NotImplementedError(f"Unsupported Assign dtype: {stmt.dtype}")
            dtype = stmt.dtype.name
            if stmt.op.name == "copy":
                # Cast directly: an intermediate f32 can change f64-to-f16 rounding.
                name = stmt.args[0]
                self.values[stmt.name] = self.convert(self.values[name], self.value_types[name], dtype)
            else:
                work_type = "f32" if dtype == "f16" else dtype
                typed_args = [self.convert(self.values[name], self.value_types[name], dtype) for name in stmt.args]
                args = [self.convert(value, dtype, work_type) for value in typed_args]
                self.values[stmt.name] = self.arithmetic(stmt.op.name, args, dtype)
            self.value_types[stmt.name] = dtype
        elif isinstance(stmt, Write) and not stmt.atomic and stmt.swizzle == "NONE":
            dtype = self.buffer_types[stmt.output]
            _, suffix, nbytes = TYPES[dtype]
            address = self.address(stmt.output, stmt.index)
            for i, name in enumerate(stmt.values):
                value = self.convert(self.values[name], self.value_types[name], dtype)
                self.emit(f"st.global.{suffix} [{address}+{nbytes * i}], {value};")
        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    # Build the complete PTX function: arguments, thread mapping, body, and declarations.
    def kernel(self, kernel, argument_order, buffer_dtypes=None):
        if kernel.serial or len(kernel.body) != 1:
            raise NotImplementedError("Expected one simple pointwise Tile")
        tile = kernel.body[0]
        if (not isinstance(tile, Tile) or len(tile.axes) != 1
                or tile.block_threads is not None or tile.aux_threads
                or tile.raster_axes is not None):
            raise NotImplementedError("Expected one simple pointwise Tile")

        self.buffer_types = dict(buffer_dtypes or {})
        for stmt in tile.body:
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

        # Tile: calculate this thread's coordinate and skip excess threads.
        block, size, thread = [self.register("r") for _ in range(3)]
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {size}, %ntid.x;")
        self.emit(f"mov.u32 {thread}, %tid.x;")
        wide_block, wide_size, wide_thread, coordinate = [self.register("rd") for _ in range(4)]
        self.emit(f"cvt.u64.u32 {wide_block}, {block};")
        self.emit(f"cvt.u64.u32 {wide_size}, {size};")
        self.emit(f"cvt.u64.u32 {wide_thread}, {thread};")
        self.emit(f"mul.lo.s64 {coordinate}, {wide_block}, {wide_size};")
        self.emit(f"add.s64 {coordinate}, {coordinate}, {wide_thread};")
        predicate = self.register("p")
        extent = tile.axes[0].extent.value
        if not isinstance(extent, int) or not 0 <= extent < 2**63:
            raise NotImplementedError("Tile extent must be a static signed-64-bit size")
        self.emit(f"setp.ge.u64 {predicate}, {coordinate}, {extent};")
        self.emit(f"@{predicate} bra DONE;")
        self.values[tile.axes[0].name] = coordinate
        self.value_types[tile.axes[0].name] = "i64"

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
    if any(arg.get("scalar") for arg in launch["args"]):
        raise NotImplementedError("Only buffer arguments are supported")
    buffer_dtypes = {arg["name"]: arg["dtype"] for arg in launch["args"]}
    print(Compiler().kernel(kernel, launch["arg_order"], buffer_dtypes), end="")


if __name__ == "__main__":
    main()
