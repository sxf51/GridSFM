# GridSFM on Windows — verification report

**Question:** can `microsoft/GridSFM` be set up and run end-to-end on this Windows machine?

**Answer: yes — all three components run.** One real, reproducible Windows blocker was found in
`power_grid/US/topology_solver_pipeline/` and fixed (line endings). Everything else works as-is.

| Component | Verdict | Evidence |
|---|---|---|
| `model/` (neural surrogate) | ✅ **Works** | 32 passed / 0 failed; real inference on 53 samples, cost MAPE 0.61% |
| `power_grid/US/viewer/` | ✅ **Works** | Served real Alabama data; 1 fixable encoding bug |
| `power_grid/US/topology_solver_pipeline/` (Julia) | ✅ **Works — Docker AND native Windows** | `✅ ALL STAGES PASSED`, exit 0, both paths |
| End-to-end (Julia → Python) | ✅ **Works** | Julia-generated scenarios consumed by the model |

GPU-bound work (fine-tuning only) is covered separately in
[COLAB_GPU_GUIDE.md](COLAB_GPU_GUIDE.md).

Machine: Windows 10.0.26200, GBK (cp936) locale, **no discrete GPU (CPU only)**,
Docker Desktop 29.6.2 running, MSYS2 at `D:\msys64`, uv 0.10.7, git 2.51.0, Julia 1.11.9.

---

## 1. `model/` — neural surrogate

### Environment (`uv`, as recommended)

`uv sync` works. Note it resolved **CPython 3.13.12** (the repo targets 3.10+ / CI on 3.12) and
still passed everything:

```powershell
cd D:\AMyUni\GridSFM\model
uv sync --extra test        # exit 0 — 164 resolved, 39 installed
uv run python -m pytest -q  # 32 passed, 6 skipped
```

Resulting core stack: `torch 2.8.0+cpu`, `torch_geometric 2.8.0.post1`, `numpy 2.5.3`,
`scipy 1.18.1`, `huggingface_hub 0.36.2`, `pytest 8.4.2`.

The pre-existing `model\.venv` (made with Anaconda's `python -m venv`, Python 3.12) also passes
identically, so **both** the plain-`venv` and `uv` paths are good:

```
python -m pytest -q   →  32 passed, 6 skipped in 19.14s   (python 3.12, anaconda venv)
uv venv pytest -q     →  32 passed, 6 skipped in 35.36s   (python 3.13, uv venv)
```

Before the checkpoint was present the suite was 20 passed / 18 skipped — the skips are
checkpoint- and OPFData-cache-gated, not Windows failures.

### Checkpoint + real inference

```powershell
hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt --local-dir checkpoints   # 58.2 MB
python examples\infer_samples.py                                                  # 65 s wall
```

```
Samples: 53 total | 53 feasible | 0 infeasible
Cost MAPE (n=53): mean=0.61%  median=0.40%  max=2.86%
V_MAE     mean = 0.0030
theta_MAE mean = 0.0210 rad (1.2049 deg)
Pg_MAE    mean = 0.0173 pu
Qg_MAE    mean = 0.0486 pu
Feas accuracy: 53/53 = 100.0%
Time: prep=11.7s  forward=35.4s (cpu)
```

These are the accuracy levels the upstream docs advertise — the model reproduces on Windows CPU.

### API surface + data loader

- `GridSFM_PG_Loader` works: `repo_type='dataset'` handled correctly, **54 regions** discovered,
  `load_model('alabama', hour='16h')` → 494 buses / 55 gens / 864 branches.
- `predict()` on `case1354_pegase.pyg.json`: `V(1354,)`, `Pg(260,)`, `Pij(1991,)`,
  `flow_edge_types=['ac_line','transformer']`, `flow_edge_counts=[1751,240]`, `feas=1.0`.
- `CycleBasisCache` disk persistence works on Windows — 53 cache blobs written to
  `C:\Users\<user>\.cache\gridsfm\cycle_basis\`.

---

## 2. `power_grid/US/viewer/`

Runs and serves real data on Windows:

```
GET /api/states                    -> 200  [{"id":"alabama","label":"Alabama",...}]
GET /api/hours                     -> 200  [{"value":"04h",...},{"value":"16h",...}]
GET /data/16h/alabama_model.json   -> 200  614,661 bytes
GET /                          -> 200  46,604 bytes (index.html)
GET /data/../../windows/win.ini    -> 404  (traversal blocked)
```

### Bug found and confirmed: emoji crash on redirected stdout

`serve.py:124` and `serve.py:133` print emoji (`⚠️`, `🗺️`). With stdout redirected (pipe, log file,
CI) Python encodes with the ANSI codepage, and on this GBK host it aborts **before binding**:

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0' in position 0
  at serve.py line 124
```

It is fine on an interactive console. Workaround: `python -X utf8 serve.py ...` or set
`PYTHONUTF8=1`. A one-line fix is to drop the emoji or pass `errors="replace"`.

---

## 3. `power_grid/US/topology_solver_pipeline/` (Julia)

### The blocker: CRLF line endings break every shell script

The repo ships **no `.gitattributes`**, and Git for Windows defaults to `core.autocrlf=true`. So a
fresh Windows clone rewrites the shell scripts to CRLF, which Linux `bash` rejects:

```
$ git ls-files --eol power_grid/US/topology_solver_pipeline/
i/lf  w/crlf  attr/     run_gen_gridsfm_data.sh
i/lf  w/crlf  attr/     docker/run_pipeline.sh
i/lf  w/crlf  attr/     integration_test_all_components.sh
```

git stores **LF**; only the working tree is CRLF. Consequences, reproduced:

```powershell
make run STATE=alabama HOUR=16h DATA_DIR=... 
# /app/docker/run_pipeline.sh: line 28: set: pipefail: invalid option name
# make: *** [Makefile:97: run] Error 2        (exit 2)

# and inside the built image:
bash -n /app/run_gen_gridsfm_data.sh              # line 54: syntax error: unexpected end of file
bash -n /app/integration_test_all_components.sh   # line 209: syntax error near '$'do\r''
bash -n /app/docker/run_pipeline.sh               # line 175: syntax error near 'done'
```

`od -c` on line 28 of the script inside the container shows `s e t - e u o p i p e f a i l \r \n`.
This also breaks the WSL path — `bash -n run_gen_gridsfm_data.sh` fails under WSL Ubuntu until the
CR is stripped.

### The fix (applied)

Added `.gitattributes` at the repo root and normalized the working tree to LF:

```gitattributes
*.sh text eol=lf
```

```powershell
# normalize the three scripts in the working tree
Get-ChildItem -Recurse -Filter *.sh | ForEach-Object {
  [IO.File]::WriteAllText($_.FullName, ([IO.File]::ReadAllText($_.FullName) -replace "`r`n","`n"))
}
```

**This changes no tracked content.** The index and HEAD blob hashes are byte-identical before and
after; `git diff --quiet` exits 0. Only the working tree changed, and it now matches what git
already stores, so a future Windows checkout will get LF for `*.sh`.

### After the fix: full pipeline passes

```powershell
make build     # exit 0 — julia:1.11-bookworm + PowerModels 0.21.5, Ipopt 1.14.0, JuMP 1.29.4
make run STATE=alabama HOUR=16h DATA_DIR=D:/.../pgdata OUTPUT_DIR=D:/.../pgout
```

```
MODEL SUMMARY — Buses: 494  Branches: 864  Generators: 55   (9501.2 MW load)
Status: LOCALLY_SOLVED   Solve time: 9.05 sec
Total cost: 198.5 k$/hr  |  Avg cost: 20.9 $/MWh
✅ Solved at L0 (Strict) — no relaxations needed
✅ Stage 2a PASSED   ✅ Stage 2b PASSED   ✅ Stage 3 PASSED
RESOLVE ok obj=198538.08 expected=198538.08 Δ=0.0000%
...
✅ ALL STAGES PASSED for alabama (16h)      exit 0
```

Artifacts produced (10 files), including `alabama_model.solvable.json`, `alabama_model.pyg.json`
and 6 perturbed scenarios (`base_unperturbed`, `costs`, `loads`, `killgen`, `derate`, `vsqueeze`).

### Native (non-Docker) local mode — also verified

Julia was **not** actually usable at first: although the juliaup/MS-Store launcher was installed,
`juliaup.json` had `"InstalledVersions": {}` and `"InstalledChannels": {}` — no Julia binary had
been downloaded (an interrupted install; stale `.tmp*`/`julia-temp-*` dirs remained), so
`julia --version` failed with *"The Julia launcher failed to figure out which juliaup channel to
use."* Installing the 1.11 channel fixed it (`juliaup add 1.11; juliaup default 1.11` →
**julia 1.11.9**, matching the pinned `Manifest.toml` and the Dockerfile's `julia:1.11`).

```powershell
make local-install   # exit 0 — 98 deps precompiled in 204 s
make local-run STATE=alabama HOUR=16h DATA_DIR=D:/.../pgdata OUTPUT_DIR=D:/.../pgout-local
```

```
✓ All packages loaded
MODEL SUMMARY — Buses: 494  Branches: 864  Generators: 55
Status: LOCALLY_SOLVED   Solve time: 13.52 sec   Avg cost: 20.9 $/MWh
RESOLVE ok obj=198538.08 expected=198538.08 Δ=0.0000%   (× 6 scenarios)
=== LOCAL-RUN EXIT=0 ===
```

Notable: `gen_perturbed_data.jl`'s `addprocs(2)` + `pmap` **multiprocessing works on Windows**
(workers 2 and 3 both ran). The audit's worry about `Xorg_libpciaccess_jll` having no Windows
artifact did **not** materialize — it installed and precompiled fine.

**Reproducibility.** The first Docker and native runs produced different perturbed objectives.
This is *not* a platform difference: `gen_perturbed_data.jl:201` derives its RNG seed from the file
path (`MersenneTwister(42 + sidx + hash((file, mode), UInt(0)))`), and the two runs used different
output directories. Julia's `hash` was confirmed stable across separate processes
(`3319930415159509266` on three consecutive runs), and re-running natively into the *same* directory
reproduced every objective exactly:

| scenario | Docker run | native run #1 | native run #2 (same path) |
|---|---|---|---|
| base_unperturbed | 198538.08 | 198538.08 | **198538.08** |
| costs_0001 | 206758.44 | 206421.11 | **206421.11** |
| derate_0001 | 200323.15 | 198538.08 | **198538.08** |
| killgen_0001 | 200519.42 | 204952.69 | **204952.69** |
| loads_0001 | *(infeasible)* | 157974.39 | **157974.39** |
| vsqueeze_0001 | 198565.63 | 198559.67 | **198559.67** |

So the pipeline is fully deterministic for a given output path, on both platforms.

### Cross-validation between the two halves of the repo

The Julia-generated scenarios were fed straight into the Python model (Docker output above;
the native output behaves the same, with the differing perturbations noted above):

| scenario | n_bus | V_min | V_max | feas |
|---|---|---|---|---|
| base_unperturbed | 494 | 0.9866 | 1.0514 | 1.000 |
| costs_0001 | 494 | 0.9867 | 1.0514 | 1.000 |
| derate_0001 | 494 | 0.9854 | 1.0514 | 1.000 |
| killgen_0001 | 494 | 0.9867 | 1.0507 | 1.000 |
| **loads_0001** | 494 | 0.9883 | 1.0573 | **0.159** |
| vsqueeze_0001 | 494 | 0.9866 | 1.0514 | 1.000 |

The pipeline labelled `loads_0001` **infeasible** (Ipopt `LOCALLY_INFEASIBLE`) and the neural
feasibility head independently assigns it 0.159 while giving 1.000 to the others — the two
components agree. The pipeline output is fully consumable by the model.

---

## 4. Remaining Windows caveats (non-blocking)

1. **Viewer emoji crash** on redirected stdout (§2) — needs `-X utf8` or a one-line code fix.
2. **`huggingface_hub` symlink warning** — Windows without Developer Mode/admin cannot create
   symlinks, so the HF cache silently degrades to duplicated files using more disk. Harmless.
3. **Pipeline needs a POSIX userland.** Docker (used above) or MSYS2/WSL + GNU Make. Git for
   Windows does *not* ship `make`; the README never says this. MSYS2's `python3` also satisfies
   `integration_test_all_components.sh:136`, which hardcodes `python3`.
4. **`Manifest.toml` is out of sync with `Project.toml`** — Julia warns
   "The project dependencies or compat requirements have changed since the manifest was last
   resolved. It is recommended to `Pkg.resolve()`". Benign, but noisy.
5. **FT tests silently skip on Windows.** `tests/test_ft.py:101,117,150` hardcode
   `/data/OPF/opfdata_pyg_train`, which resolves to `<current drive>:\data\...`, so 6 tests
   auto-skip. Coverage loss, not a failure.
6. **Locale-dependent JSON reads.** Several `open()` calls omit `encoding=`; a non-ASCII JSON would
   raise `UnicodeDecodeError` under GBK. All 53 shipped samples (172 MB) were verified **pure
   ASCII**, so this does not bite today.
7. `serve.py` prints mixed separators in its hint (`data_dir/16h/...`) — cosmetic.

---

## 5. Not tested

- **Fine-tuning** (`examples/finetune_opfdata_case6470.ipynb`) — needs OPFData plus a GPU. This
  machine has **no discrete GPU**. This is the *only* genuinely GPU-bound part of the repo;
  see [COLAB_GPU_GUIDE.md](COLAB_GPU_GUIDE.md) for a copy-paste Colab recipe plus a ready-to-run
  smoke test at [`model/colab_finetune_smoke.py`](model/colab_finetune_smoke.py).
- **`examples/opfdata.py` on a full split** — not run here (large OPFData download). It is plain
  inference, so CPU works; the guide's Cell 7 runs it on a Colab GPU.

Everything else in the repo has now been executed on this machine.

---

## 6. Exact reproduction

```powershell
# --- model/ (uv) ---
cd D:\AMyUni\GridSFM\model
uv sync --extra test
uv run python -m pytest -q
uv run hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt --local-dir checkpoints
uv run python examples\infer_samples.py

# --- viewer/ ---
cd D:\AMyUni\GridSFM\power_grid\US\viewer
python -X utf8 serve.py --data-dir <dir with 16h/ and 04h/>

# --- Julia pipeline (Docker) ---
cd D:\AMyUni\GridSFM\power_grid\US\topology_solver_pipeline
$env:MSYS_NO_PATHCONV="1"
make build
make run STATE=alabama HOUR=16h DATA_DIR=<data dir> OUTPUT_DIR=<out dir>

# --- Julia pipeline (native, no Docker) ---
juliaup add 1.11; juliaup default 1.11     # once, if julia --version fails
make local-install                          # ~3 min, 98 deps precompiled
make local-run STATE=alabama HOUR=16h DATA_DIR=<data dir> OUTPUT_DIR=<out dir>
```

## 7. Files changed / added in this checkout

| File | Change |
|---|---|
| `.gitattributes` | **New.** `*.sh text eol=lf` — prevents CRLF regression on Windows clones |
| `power_grid/.../docker/run_pipeline.sh` | Working tree CRLF → LF (no tracked-content change) |
| `power_grid/.../integration_test_all_components.sh` | Working tree CRLF → LF (no tracked-content change) |
| `power_grid/.../run_gen_gridsfm_data.sh` | Working tree CRLF → LF (no tracked-content change) |
| `WINDOWS_COMPATIBILITY_REPORT.md` | **New.** This report |
| `COLAB_GPU_GUIDE.md` | **New.** GPU parts + Colab recipe |
| `OPFDATA_GUIDE.md` | **New.** Where OPFData lands, what is inside it, how to use it |
| `model/colab_finetune_smoke.py` | **New.** Ready-to-run GPU fine-tuning smoke test |
| `model/uv.lock` | Untracked, pre-existing before this session (uv resolution lock) |

Scratch/verification artifacts (safe to delete): `D:\AMyUni\gridsfm-winverify\`,
the Docker image `topo_solver_pipe:latest`, and `model\checkpoints\gridsfm_open_v1.1.pt` (58 MB,
needed to run inference/tests).
