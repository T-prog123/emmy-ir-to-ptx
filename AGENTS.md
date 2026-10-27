# Agent instructions

## Rules

- **NEVER touch Git without explicit user authorization for that specific action.** Do not run Git commands, edit `.git`, commit, push, change remotes/visibility, or attempt ownership/permission repairs on your own.
- This is a learning project. Keep code readable and explanations concise, with concrete examples and current source line numbers. Use short comments.
- Preserve existing capabilities. Avoid unsolicited scope expansion and unrelated refactors.
- Leave organizer problems, custom fixtures and historical snapshots unchanged unless asked to modify them.

## Contract

- All translation lives in `compile.py`: `python compile.py path/to/kernel.ir.json > out.ptx`.
- Python 3.12+, `emmy-ml==0.3.11`. Load typed objects with `Graph.from_dict`; do not parse repr strings yourself.
- Output only PTX to stdout; errors go to stderr. Current target: PTX 8.0, sm_89, 64-bit addresses.
- Read neighboring `meta.json` for entry-matched argument order, buffer types/shapes and prescribed launch. Constants and outputs also arrive as buffer addresses.
- Implement the lowering ourselves. Do not delegate translation to Emmy's renderer/compiler, nvcc or another compiler. Emmy's loader is allowed.
- Compiler runtime must not depend on sibling workspace directories.

## Architecture and scope

- `main()` loads files; `kernel()` prepares buffers, Tile coordinates and declarations; `statement()` dispatches computations, nested control flow and collectives.
- `values` maps names to registers; `value_types` tracks their types. `buffers`, `buffer_types` and `buffer_shapes` describe arrays. Counters allocate virtual register names, not hardware registers.
- `carried` stores stable f32 accumulator registers: seed before the loop, update each iteration, preserve afterward.
- Pointwise: f16/f32/f64/i32/i64 storage/basic arithmetic and conversions. Half arithmetic works through f32 and rounds after each operation. Exp supports f16/f32; sqrt supports floating types. Reject unsupported combinations clearly.
- Signed 64-bit index arithmetic uses floor division/remainder; integer value division truncates. Intermediates and byte offsets must fit signed 64 bits.
- Static shapes, flat or full-rank row-major accesses, launch x dimensions only. Grouped accesses are scalarized.
- Reductions: scalar f32 additive state, array sums/means and row sums. One complete cooperative block per result, 32-1024 threads; innermost Tile axis matches `block_threads`.
- Shared scratch is static. `TreeHalve` must broadcast the result to every thread. Use one writer per shared slot; keep block barriers outside divergent control flow.
- No other reduction types/ops, cross-block combines, partial warps, named/warp barriers, auxiliary/raster schedules, ordinary `Loop`, matmul, async copies or atomics.

## Workspace and validation

These development resources may be absent in a standalone clone:

- `.venv/Scripts/python.exe`: existing Emmy environment; do not upgrade Emmy automatically.
- `../emmy-hackathon-2026/`: organizer docs, fixtures, runner API and `submit.py`.
- `../emmy-hackathon-personal-work/validate_category_1.py --stage 4` and `validate_reductions.py --rows`: validation helpers in personal-work. Both accept `--compiler PATH`, `--report PATH`, `--local-only`.
- Helpers calculate independent NumPy answers and execute on the remote GPU. Preserve result types: exact integer comparisons, suitable floating tolerances. Existing word transport avoids JSON precision/NaN problems.
- V7 reports in personal-work: `category_1_validation_v7.json` (138 GPU passes, 2 expected rejections), `reduction_validation_v7.json` (46 GPU passes, 13 expected rejections).
- Old reports are not fresh runs or proof of complete hidden-set coverage. Translation/assembly checks are not GPU correctness tests. If the runner is unavailable, record numerical validation as pending.

## Checkpoints

- Baseline: V7. Snapshots live in `../emmy-hackathon-personal-work/backup_versions_compiler/`.
- After a significant implemented milestone, verify it, save a full `compile_vN.py` using the next unused number, and add **one short line** to `compiler_backup_readme.md` with changes and actual results.
- Preserve previous snapshots. Update README and personal-work limits when support changes. Documentation-only edits need no compiler checkpoint.
