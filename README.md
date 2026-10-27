# Emmy IR to PTX

A small learning compiler for category 1 pointwise kernels and f32 reductions.
All translation code is in `compile.py`. Requires Python 3.12 or newer.

## Run

From this directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe compile.py ..\emmy-hackathon-2026\problems\p00_scalar_add\k_add_pointwise.ir.json | Set-Content -Encoding ascii out.ptx
```

The input's neighbouring `meta.json` supplies argument order, buffer types and shapes.
The compiler writes PTX to stdout; it does not allocate arrays or run GPU code.

## Read the code

- `register`: hand out temporary register names, starting at zero.
- `expression`: translate signed 64-bit indices (+, -, *, floor division, remainder).
- `floor_divmod`: adjust PTX's truncating division to Emmy's floor semantics.
- `address`: multiply the index by the element byte size, then add its buffer address.
- `convert` / `arithmetic`: handle typed values and one scalar operation.
- `statement`: translate loads, arithmetic, writes, and nested control flow.
- `strided_loop`: seed and update running sums while threads stride the input.
- `warp_shuffle` / `tree_halve`: combine partial sums and broadcast the result.
- `kernel`: add parameters, Tile coordinates, guards, and declarations.
- `main`: load the IR and metadata, then print the generated PTX.

For clarity, address calculations are repeated and `+ 0` is retained.
Grouped loads and writes become individual typed loads and stores.
There is no optimization or register reuse logic.

## Current scope

One static Tile, with flat or full-rank row-major accesses and launch x dimensions.
Multidimensional accesses need buffer shapes; arbitrary strides/views are unsupported.
Storage and basic arithmetic support f16, f32, f64, i32 and i64. Conversions use
the declared types; half arithmetic rounds back after each IR operation.
Arithmetic support: add, subtract, multiply, divide, minimum, maximum, ReLU,
exp and sqrt. Exp supports f16/f32; sqrt supports all three float types.
Copy/cast assignments are supported. Other combinations and constructs raise an error.
Integer value division truncates toward zero; index division uses floor semantics.
Indices, their intermediate calculations, and byte offsets must fit signed 64 bits.
The prescribed launch must cover the Tile. This is tested support, not certification
of the hidden evaluation set.

Category 2 support starts with f32 additive reductions: array sums, means, and
row sums (r01-r04). Cooperative Tiles use complete blocks of 32-1024 threads,
with one innermost cooperative axis. Each block reduces its own row.
Shared scratch is static; grouped accesses remain scalarized. Conditions support
integer comparisons/logical expressions. Loop steps are positive integer constants.
Only scalar f32 combines and flat per-warp trees are supported. Partial warps,
conditional barriers, named/warp barriers, auxiliary/raster schedules, other reduce
operations/types, ordinary Loop, async copies and atomics are rejected.

Checkpoint v2 passed 22 numerical checks on the remote GPU, including p00-p09
and special floating-point values. Saved versions and results are in the personal
work directory.
Checkpoint v3 passed 83 distinct numerical cases, including cross-type conversions
and values requiring double/integer precision. The validator uses exact 32-bit
transport words for i64 because the runner's ordinary JSON path rounds large values.
Checkpoint v4 passed 90 GPU checks, adding repeated-input indexing, negative
division/remainder, and large intermediate index calculations.
Checkpoint v5 passed 132 GPU numerical cases (all p00-p19 plus focused probes),
and two checks that unsupported combinations are rejected. It fixes half signed-zero
ties and mixed-type half rounding discovered during broader validation.
Checkpoint v6 passed 35 reduction and 138 category 1 GPU checks. Checkpoint v7
passed 46 reduction and 138 category 1 GPU checks, including r01-r04 and p20-p25;
15 unsupported-case checks were rejected as expected. Current compile.py is v7.

Reduction checks use `../emmy-hackathon-personal-work/validate_reductions.py --rows`
with the same .venv. It compares every output against independent NumPy answers
on the remote GPU; `--local-only` performs translation/rejection checks only.

## Test on the hackathon runner

Install the provided testing script's dependencies into the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy requests
.\.venv\Scripts\python.exe ..\emmy-hackathon-2026\submit.py p00_scalar_add --compiler compile.py
```

Other supported originals include p01-p03 and r01-r04.

## Validate the extended compiler

From this directory, using the same environment:

```powershell
.\.venv\Scripts\python.exe ..\emmy-hackathon-personal-work\validate_category_1.py --stage 4 --report ..\emmy-hackathon-personal-work\category_1_validation_latest.json
.\.venv\Scripts\python.exe ..\emmy-hackathon-personal-work\validate_reductions.py --rows --report ..\emmy-hackathon-personal-work\reduction_validation_latest.json
```

The helper computes independent NumPy answers in memory, preserves result types,
and calls the remote GPU API. It does not add files to problem directories.
It uses 32-bit transport words for i64 and special float values to avoid the runner's
JSON precision/NaN limitations; the GPU still accesses the original typed bytes.

`--local-only` checks translation without contacting the GPU; it is not a numerical
test. `--only f16` selects matching case names. `--stage 1`, `2`, or `3` runs the
corresponding cumulative subset; `--compiler PATH` tests a saved snapshot.

Snapshots v1-v7 and their short change log live in the personal-work backup directory.

