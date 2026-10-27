# Emmy IR to PTX

A Python compiler that translates Emmy Kernel IR directly into PTX. Requires Python 3.12+.

## Run

```bash
python -m pip install -r requirements.txt
python compile.py path/to/kernel.ir.json > out.ptx
```

Keep the supplied `meta.json` beside the IR file; it defines arguments, types, shapes and launch geometry.

## Support

- **Pointwise:** f16/f32/f64/i32/i64 arithmetic, conversions, grouped loads/stores and 64-bit indexing. Exp supports f16/f32; sqrt supports floating types.
- **Reductions:** f32 sums and means of 1-D arrays or matrix rows, with one complete thread block per result.
- **Layout:** static shapes, flat or row-major addressing, launch x dimensions only.

Other reduction operations/types, cross-block reductions, matmul, async copies and atomics are unsupported.
