"""A small learning compiler for the pointwise examples p00-p03."""

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import (
    Assign, BinaryExpr, KernelOp, Literal, Load, Tile, Var, Write,
)


class Compiler:
    # Prepare the output list, register counters, and name-to-register mappings.
    def __init__(self):
        self.lines = []
        self.counts = {"p": 0, "r": 0, "rd": 0, "f": 0}
        self.values = {}       # IR value name -> register name
        self.buffers = {}      # Buffer name -> register holding its address

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
            return self.values[expr.name]
        # case 2, the variable is an int, ie directly a value
        if isinstance(expr, Literal) and expr.dtype == "int":
            return str(expr.value)
        # case 3: computation has to be done to get the expression
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*"):
            left = self.expression(expr.left)
            right = self.expression(expr.right)
            result = self.register("r")
            operation = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32"}[expr.op]
            self.emit(f"{operation} {result}, {left}, {right};")
            return result
        raise NotImplementedError(f"Unsupported index expression: {expr}")

    # Generate PTX to find an array element's address and return its register name.
    def address(self, buffer, indices):
        if len(indices) != 1:
            raise NotImplementedError("Only one-dimensional indices are supported")
        
        # we do the necessary computation on the index ("(a0 * 2) + 0")
        index = self.expression(indices[0])

        offset = self.register("rd")
        address = self.register("rd")
        self.emit(f"mul.wide.s32 {offset}, {index}, 4;")
        self.emit(f"add.u64 {address}, {self.buffers[buffer]}, {offset};")
        return address

    # Generate PTX for one Load, Assign, or Write from the Tile body.
    def statement(self, stmt):
        # in this case stms is a "Load", with input, names, index,
        # and type.name
        if isinstance(stmt, Load) and stmt.dtype.name == "f32":
            # we give input ("x0") and index("(a0 * 2) + 0") to
            # do the necessary work to compute the adress needed for 
            # the global memory load 
            address = self.address(stmt.input, stmt.index)
            for i, name in enumerate(stmt.names):
                result = self.register("f")
                self.emit(f"ld.global.f32 {result}, [{address}+{4 * i}];")
                self.values[name] = result

        elif isinstance(stmt, Assign) and stmt.dtype.name == "f32":
            args = [self.values[name] for name in stmt.args]
            result = self.register("f")
            if stmt.op.name == "add":
                self.emit(f"add.rn.f32 {result}, {args[0]}, {args[1]};")
            elif stmt.op.name == "multiply":
                self.emit(f"mul.rn.f32 {result}, {args[0]}, {args[1]};")
            # relu corresponds to a max with 0 ("partie positive")
            elif stmt.op.name == "relu":
                self.emit(f"max.f32 {result}, {args[0]}, 0.0;")
            else:
                raise NotImplementedError(f"Unsupported arithmetic: {stmt.op.name}")
            self.values[stmt.name] = result

        elif (isinstance(stmt, Write) and stmt.value_dtype.name == "f32"
              and not stmt.atomic and stmt.swizzle == "NONE"):
            address = self.address(stmt.output, stmt.index)
            for i, name in enumerate(stmt.values):
                self.emit(f"st.global.f32 [{address}+{4 * i}], {self.values[name]};")

        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    # Build the complete PTX function: arguments, thread mapping, body, and declarations.
    def kernel(self, kernel, argument_order):
        if kernel.serial or len(kernel.body) != 1:
            raise NotImplementedError("Expected one simple pointwise Tile")
        tile = kernel.body[0]
        if (not isinstance(tile, Tile) or len(tile.axes) != 1
                or tile.block_threads is not None or tile.aux_threads
                or tile.raster_axes is not None):
            raise NotImplementedError("Expected one simple pointwise Tile")

        # Read the array addresses passed by the launcher
        # (the 3 input adresses/paramter in p00)
        for i, name in enumerate(argument_order):
            address = self.register("rd")
            self.emit(f"ld.param.u64 {address}, [arg_{i}];")
            self.emit(f"cvta.to.global.u64 {address}, {address};")
            self.buffers[name] = address

        # Tile: calculate this thread's coordinate and skip excess threads.
        block, size, thread, coordinate = [self.register("r") for _ in range(4)]
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {size}, %ntid.x;")
        self.emit(f"mov.u32 {thread}, %tid.x;")
        self.emit(f"mad.lo.u32 {coordinate}, {block}, {size}, {thread};")
        predicate = self.register("p")
        self.emit(f"setp.ge.u32 {predicate}, {coordinate}, {tile.axes[0].extent.value};")
        self.emit(f"@{predicate} bra DONE;")
        self.values[tile.axes[0].name] = coordinate

        # Translate the IR KernelOp's tile to ptx.
        for stmt in tile.body:
            self.statement(stmt)
        self.lines.append("DONE:")
        self.emit("ret;")

        # We now know how many register names the generated instructions use.
        types = {"p": "pred", "r": "b32", "rd": "b64", "f": "f32"}
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
    print(Compiler().kernel(kernel, launch["arg_order"]), end="")


if __name__ == "__main__":
    main()
