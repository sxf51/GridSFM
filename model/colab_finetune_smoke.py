"""GridSFM fine-tuning smoke test — run this on a GPU (e.g. Google Colab).

Mirrors the README's "n=16 round finishes in a couple of minutes and is a useful
smoke test that FT is wired up correctly".

Run from the `model/` directory:
    python colab_finetune_smoke.py

Prerequisites:
    pip install -e ".[test,notebook]"
    hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt --local-dir checkpoints

The first run downloads the OPFData cache for the chosen case (this is the slow
part and it is NOT reduced by n_graphs -- see gridsfm/opfdata_train.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# --- configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE if (HERE / "gridsfm").is_dir() else Path.cwd()
CKPT = Path(os.environ.get("GRIDSFM_CKPT", MODEL_DIR / "checkpoints" / "gridsfm_open_v1.1.pt"))
OPFDATA_ROOT = os.environ.get("OPFDATA_ROOT", "/content/opfdata")
CASE = os.environ.get("GRIDSFM_CASE", "pglib_opf_case500_goc")
N_GRAPHS = int(os.environ.get("GRIDSFM_N_GRAPHS", "16"))
EPOCHS = int(os.environ.get("GRIDSFM_EPOCHS", "2"))
# ---------------------------------------------------------------------------

sys.path.insert(0, str(MODEL_DIR))

from gridsfm import (  # noqa: E402
    load_model, OPFDataAdapterDataset, SyntheticMixedDataset, finetune_opfdata,
)
from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_  # noqa: E402
from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("!! CUDA is not available -- this will run on CPU and be VERY slow.")
        print("   On Colab: Runtime -> Change runtime type -> T4 GPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only"
    print(f"device : {device}  ({name})")
    print(f"torch  : {torch.__version__}")

    if not CKPT.exists():
        print(f"\n!! checkpoint not found at {CKPT}\n"
              f"   hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt "
              f"--local-dir {CKPT.parent}")
        return 1
    print(f"ckpt   : {CKPT}  ({CKPT.stat().st_size / 1e6:.1f} MB)")

    model = load_model(str(CKPT), device=device)

    # REQUIRED transform: cycle basis + Hodge PE. Without it the model's input
    # projections get no `cycle` / `branch_*` features and the forward silently
    # degrades (HodgePE no-ops, GNN conv -> zero output).
    cb_cache = CycleBasisCache()
    pe_cache = LaplacianFactorizationCache()

    def transform(data):
        prepare_for_grid_transformer_(data, cache=cb_cache)
        attach_pe_features_(data, cache=pe_cache)
        return data

    print(f"\nloading OPFData case={CASE} root={OPFDATA_ROOT} "
          f"(first call downloads the shard cache; this can take a while)")
    base = OPFDataAdapterDataset(
        root=OPFDATA_ROOT, case_name=CASE, variant="fulltop",
        split="train", n_graphs=N_GRAPHS,
    )
    print(f"  train graphs available: {len(base)}")

    train_ds = SyntheticMixedDataset(base, infeas_prob=0.3, seed=42, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    print(f"\nfine-tuning for {EPOCHS} epoch(s) on {len(train_ds)} graphs ...")
    log = finetune_opfdata(model, train_loader, epochs=EPOCHS, lr=1e-4)

    print(f"\n{'epoch':>5}  {'train_loss':>12}  {'iters':>6}  {'epoch_s':>8}")
    print("-" * 38)
    for e in log:
        print(f"{e['epoch']:>5}  {e['train_loss']:>12.6f}  "
              f"{e['n_train_iters']:>6}  {e['epoch_s']:>8.1f}")

    losses = [e["train_loss"] for e in log if e["n_train_iters"] > 0]
    if not losses:
        print("\nFAILED: no batch produced a finite loss.")
        return 1
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"\nOK: train_loss decreased {losses[0]:.6f} -> {losses[-1]:.6f} "
              f"(fine-tuning is wired up correctly)")
    else:
        print(f"\nWARNING: train_loss did not decrease ({losses[0]:.6f} -> {losses[-1]:.6f}). "
              f"With only {N_GRAPHS} graphs / {EPOCHS} epochs this can be normal; "
              f"raise GRIDSFM_N_GRAPHS/GRIDSFM_EPOCHS before concluding anything.")
    print("GPU fine-tuning smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
