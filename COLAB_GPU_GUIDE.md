# GridSFM — GPU parts and how to run them on Google Colab

Companion to [WINDOWS_COMPATIBILITY_REPORT.md](WINDOWS_COMPATIBILITY_REPORT.md).
This machine has **no discrete GPU**, so everything below was either verified on CPU here or is
marked **NOT TESTED — needs a GPU**.

## 1. What actually needs a GPU

Good news: the `gridsfm` package is **device-agnostic** — a repo-wide grep found no hardcoded
`.cuda()`, no `autocast`, no `GradScaler`. Every device is derived from the model
(`next(model.parameters()).device`), and every CLI example uses the same safe pattern:

```python
device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
```

So **no code changes are needed on Colab** — pass `--gpu 0` and it uses the GPU; on a CPU box it
silently falls back.

| Task | GPU? | Status here |
|---|---|---|
| `python -m pytest -q` | No | ✅ 32 passed on CPU |
| `examples/infer_samples.py` (53 samples) | **Optional** (speed only) | ✅ ran on CPU: 0.61% MAPE, 35 s forward |
| `examples/predict_to_viewer.py` | Optional | ✅ `--help` + import verified |
| `predict()` / `model(batch)` in your own code | Optional | ✅ verified on CPU |
| `power_grid/US/viewer/` | No | ✅ verified on Windows |
| `power_grid/.../topology_solver_pipeline/` (Julia) | **No — never** | ✅ Ipopt is a CPU solver; full pipeline passed |
| **`examples/opfdata.py`** on a full split | **Recommended** | ⚠️ Not run — needs the OPFData download |
| **Fine-tuning** (`finetune_opfdata_case6470.ipynb`) | **YES — required** | ❌ **NOT TESTED** |

**Bottom line:** exactly one thing on this repo genuinely needs a GPU — **fine-tuning**. Per its own
README the reference notebook takes *"roughly an hour on a single GPU"*; on this CPU-only machine it
would be many hours to days.

---

## 2. Colab recipe (copy-paste cells)

### Cell 1 — confirm you got a GPU

```python
!nvidia-smi
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU"
```

### Cell 2 — get the code

```python
!git clone https://github.com/microsoft/GridSFM.git
%cd /content/GridSFM
!git log --oneline -1
%cd /content/GridSFM/model
```

### Cell 3 — install

```python
!pip install -q -e ".[test,notebook]"
```

> **Watch this step.** The pin is `torch>=2.6,<2.9`. Colab ships its own CUDA torch build which may
> be outside that range, in which case pip **replaces it** with 2.8.0 (a ~2–3 GB CUDA wheel — fine,
> just slow). Re-run the `torch.cuda.is_available()` check afterwards to be sure you still have CUDA.

### Cell 4 — the checkpoint (58 MB)

```python
!hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt --local-dir checkpoints
```

### Cell 5 — full test suite (proves the install)

```python
!python -m pytest -q
```

Expected, matching what was measured on Windows CPU:
**`32 passed, 6 skipped`** — the 6 skips are OPFData-cache-gated.

### Cell 6 — GPU inference sanity check

```python
!python examples/infer_samples.py --gpu 0
```

Expect the same accuracy as the CPU run (mean cost MAPE ≈ 0.61%, feasibility 53/53) with a much
shorter forward time. This confirms CUDA is actually being used.

### Cell 7 — OPFData evaluation (optional; needs a big download)

```python
!python examples/opfdata.py --case pglib_opf_case500_goc --root /content/opfdata \
    --gpu 0 --limit 256
```

`OPFDataset` auto-downloads on first use into `--root`. Start with `--limit`; drop it for the full
split once you know it fits. This script's `--num-groups` already defaults to **1** (1.15 GB for
`case500_goc`). Cases with <500 buses are out of distribution — results are meaningless.

### Cell 8 — **fine-tuning smoke test** (the actual GPU work)

The smoke test is already in the clone at [`model/colab_finetune_smoke.py`](model/colab_finetune_smoke.py)
(Cell 2 put you in `model/`, so it is right there). Or paste it in yourself:

```python
!python colab_finetune_smoke.py
```

This mirrors the README's *"n=16 round finishes in a couple of minutes and is a useful smoke test
that FT is wired up correctly"* — 16 training graphs, 2 epochs. If `train_loss` decreases and
`val_cost_mape` is a real number, fine-tuning works.

### Cell 9 — the full fine-tuning study

Open `examples/finetune_opfdata_case6470.ipynb` in Colab (File → Open notebook → Upload, or open it
straight from the cloned repo). **You must edit one line:**

```python
# original — a Linux path that does not exist on Colab
OPFDATA_ROOT = '/data/OPF/opfdata_pyg_train'
# change to:
OPFDATA_ROOT = '/content/opfdata'
```

The notebook derives `FULLTOP_DIR`/`N1_DIR` from it, auto-detects CUDA
(`torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`), and calls
`torch.cuda.empty_cache()` between rounds. Nothing else needs changing. Expect ~1 h for the
`case6470_rte` study on a single GPU; the `n=16` round alone is a couple of minutes.

---

## 3. Practical notes for Colab

- **Disk.** Colab gives ~78 GB (`/content`). Sizes were measured against the real bucket —
  **`num_groups=1` costs**: `case500_goc` 1.15 GB, `case6470_rte` **12.9 GB**, `case10000_goc`
  18.5 GB (per shard, and the N-1 release is a second copy of the same size). The FT notebook uses
  `case6470_rte` fulltop **plus** n-1 ≈ **26 GB** before its own processed cache — it fits, but plan
  for it. Add ~4.4 min of *CPU* processing per 15,000 graphs on this class of machine (measured on
  case14; bigger cases take longer). Note the warning in `gridsfm/opfdata_train.py`: `n_graphs=16`
  does **not** make the download lighter — the first call still fetches and decodes the full
  `num_groups * 15000`-graph cache. Download once, then it is cached for the session.
  Full details in [OPFDATA_GUIDE.md](OPFDATA_GUIDE.md).
- **Always pass `num_groups=1` explicitly.** PyG's own default is **20**, which for `case500_goc`
  means 23 GB and for `case6470_rte` 258 GB. The GridSFM wrappers default to 1; a bare
  `OPFDataset(root=...)` does not.
- **`num_workers`.** Keep PyG `DataLoader(num_workers=0)` (the repo's own tests do). Colab's
  default `fork` plus big graphs makes workers expensive.
- **Persistence.** `/content` dies with the runtime; mount Drive or re-download the checkpoint each
  session. The checkpoint is only 58 MB, so re-downloading is usually simpler.
- **uv instead of pip** (if you prefer, matching the Windows setup):
  ```python
  !pip install -q uv
  !uv pip install --system -q -e ".[test,notebook]"
  ```
- **The Julia pipeline does not belong on Colab** — Ipopt is CPU-only and the topology solver is
  I/O- and solve-bound, not GPU-bound. Run it on the Windows machine (verified working) or any CPU box.

---

## 4. What I could and could not verify

| | Verified on this machine (CPU) | Needs Colab |
|---|---|---|
| Install + tests | ✅ 32 passed / 0 failed | — |
| Inference accuracy | ✅ 53 samples, 0.61% MAPE | — |
| Julia pipeline | ✅ both Docker **and** native Windows, exit 0 | — |
| Viewer | ✅ real Alabama data | — |
| `examples/opfdata.py` | ❌ not run (large download) | ✅ Cell 7 |
| **Fine-tuning** | ❌ **not run (no GPU)** | ✅ Cells 8–9 |

Nothing about Windows blocks the GPU paths: the Colab instructions above are ordinary
Linux + CUDA usage, and the Windows-specific issues found (CRLF shell scripts, viewer emoji
encoding) do not touch the Python model package at all.
