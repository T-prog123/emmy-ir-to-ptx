# Emmy IR to PTX

A small learning compiler, starting with the pointwise examples p00-p03.
All translation code is in `compile.py`. Requires Python 3.12 or newer.

## Run

From this directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe compile.py ..\emmy-hackathon-2026\problems\p00_scalar_add\k_add_pointwise.ir.json | Set-Content -Encoding ascii out.ptx
```

The input's neighbouring `meta.json` supplies the argument order.
The compiler writes PTX to stdout; it does not allocate arrays or run GPU code.

## Read the code

- `register`: hand out temporary register names, starting at zero.
- `expression`: translate integer index expressions (variables, literals, +, -, *).
- `address`: multiply a float index by four bytes, then add its buffer address.
- `statement`: translate loads, arithmetic, and writes.
- `kernel`: add parameters, thread numbering, the Tile guard, and declarations.
- `main`: load the IR and metadata, then print the generated PTX.

For clarity, address calculations are repeated and `+ 0` is retained.
Grouped loads and writes become individual float loads and stores.
There is no optimization or register reuse logic.

## Current scope

One-dimensional, unswizzled f32 pointwise kernels with one simple Tile.
Arithmetic support: add, multiply, and ReLU. Other constructs raise an error.
This is a starting implementation for the open p00-p03 examples, not complete
coverage of hidden category 1 problems. Reductions are not implemented yet.

## Test on the hackathon runner

Install the provided testing script's dependencies into the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy requests
.\.venv\Scripts\python.exe ..\emmy-hackathon-2026\submit.py p00_scalar_add --compiler compile.py
```

Repeat with `p01_vector_add`, `p02_scale_shift`, and `p03_relu_mul`.

