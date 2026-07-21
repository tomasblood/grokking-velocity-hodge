# Databricks notebook source
# MAGIC %md
# MAGIC # Grokking Training: Reproducible 1-Head Transformer
# MAGIC
# MAGIC Trains the modular addition transformer used by the Grokking chapter and
# MAGIC writes the activation snapshots consumed by the analysis notebooks.

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

sys.path.insert(0, os.environ["THESIS_SHARED_DIR"])
from runtime import configure_grokking_runtime, ensure_dir, notebook_param, write_json

GROKKING = configure_grokking_runtime()

P = int(notebook_param("GROKKING_P", "113"))
D_MODEL = int(notebook_param("GROKKING_D_MODEL", "128"))
N_EPOCHS = int(notebook_param("GROKKING_N_EPOCHS", "25000"))
DATA_SEED = int(notebook_param("GROKKING_DATA_SEED", "598"))
N_SUB = int(notebook_param("GROKKING_N_SUB", "500"))
SAVE_EVERY = int(notebook_param("GROKKING_SAVE_EVERY", "500"))
TRAIN_FRAC = float(notebook_param("GROKKING_TRAIN_FRAC", "0.3"))
PROBE_SEED = int(notebook_param("GROKKING_PROBE_SEED", "42"))
FORCE = notebook_param("GROKKING_FORCE", "0").strip().lower() in {"1", "true", "yes"}

TMP_DIR = ensure_dir(Path(notebook_param("GROKKING_TMP_DIR", "/tmp/grokking_acts_v6")))
OUT_DIR = ensure_dir(Path(notebook_param("GROKKING_ACTIVATION_DIR", str(GROKKING.activation_dir))))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

existing_outputs = list(OUT_DIR.glob("act_*.npy"))
assert not (existing_outputs and not FORCE), f"{OUT_DIR} already contains activation snapshots"
if FORCE:
    for path in [*OUT_DIR.glob("act_*.npy"), OUT_DIR / "gt_labels.npy", OUT_DIR / "training.json"]:
        if path.exists():
            path.unlink()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "activation_dir": str(OUT_DIR),
    "tmp_dir": str(TMP_DIR),
    "p": P,
    "d_model": D_MODEL,
    "n_epochs": N_EPOCHS,
    "data_seed": DATA_SEED,
    "n_sub": N_SUB,
    "save_every": SAVE_EVERY,
    "device": DEVICE,
    "force": FORCE,
}, indent=2))

# COMMAND ----------

torch.manual_seed(DATA_SEED)
a = torch.arange(P).unsqueeze(1).repeat(1, P).flatten()
b = torch.arange(P).unsqueeze(0).repeat(P, 1).flatten()
labels = (a + b) % P
dataset = torch.stack([a, b, labels], dim=1)
perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(DATA_SEED))
dataset = dataset[perm]
n_train = int(TRAIN_FRAC * len(dataset))
train_data, val_data = dataset[:n_train], dataset[n_train:]

rng = np.random.default_rng(PROBE_SEED)
sub_idx = rng.choice(P * P, N_SUB, replace=False)
all_input = dataset[sub_idx, :2]
gt_labels = dataset[sub_idx, 2].numpy()
np.save(TMP_DIR / "gt_labels.npy", gt_labels)

print(f"Train: {n_train}; validation: {len(val_data)}; probe: {N_SUB}")

# COMMAND ----------

cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=D_MODEL,
    d_head=D_MODEL,
    n_heads=1,
    d_mlp=D_MODEL * 4,
    d_vocab=P,
    n_ctx=3,
    act_fn="relu",
    normalization_type=None,
)
model = HookedTransformer(cfg)

if DEVICE == "cuda":
    model = model.cuda()
    train_data = train_data.cuda()
    val_data = val_data.cuda()
    all_input = all_input.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# COMMAND ----------

train_losses = []
val_accs = []
saved_epochs = []
t0 = time.time()

for epoch in range(N_EPOCHS + 1):
    logits = model(train_data[:, :2])[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, train_data[:, 2])

    if epoch % SAVE_EVERY == 0:
        train_loss = loss.item()
        with torch.no_grad():
            val_acc = (model(val_data[:, :2])[:, -1, :].argmax(-1) == val_data[:, 2]).float().mean().item()
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        saved_epochs.append(epoch)

        with torch.no_grad():
            _, cache = model.run_with_cache(all_input)
            act = cache["blocks.0.mlp.hook_post"][:, -1, :].cpu().numpy().astype(np.float64)
        np.save(TMP_DIR / f"act_{epoch}.npy", act)
        del cache, act
        gc.collect()

        if epoch % 5000 == 0:
            print(f"{epoch}: loss={train_loss:.4f}; val_acc={val_acc:.4f}; elapsed={time.time() - t0:.0f}s")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training finished in {time.time() - t0:.0f}s; final val_acc={val_accs[-1]:.4f}")

sample = np.load(TMP_DIR / "act_0.npy")
expected_shape = (N_SUB, D_MODEL * 4)
assert sample.shape == expected_shape, f"Expected {expected_shape}; got {sample.shape}"

# COMMAND ----------

training_meta = {
    "config": {
        "P": P,
        "d_model": D_MODEL,
        "n_heads": 1,
        "d_head": D_MODEL,
        "d_mlp": D_MODEL * 4,
        "n_epochs": N_EPOCHS,
        "save_every": SAVE_EVERY,
        "data_seed": DATA_SEED,
        "model_seed": "torch.manual_seed(598); no HookedTransformerConfig seed",
        "n_sub": N_SUB,
        "train_frac": TRAIN_FRAC,
        "probe_seed": PROBE_SEED,
        "transformer_lens_version": "2.17.0",
        "extraction": "blocks.0.mlp.hook_post at final sequence position",
    },
    "saved_epochs": saved_epochs,
    "train_losses": train_losses,
    "val_accs": val_accs,
}

write_json(TMP_DIR / "training.json", training_meta)

for epoch, acc in zip(saved_epochs, val_accs):
    if acc > 0.95:
        print(f"Generalisation at epoch {epoch} (val_acc={acc:.4f})")
        break

# COMMAND ----------

for path in TMP_DIR.iterdir():
    if path.is_file():
        shutil.copy2(path, OUT_DIR / path.name)

verified = np.load(OUT_DIR / "act_0.npy")
print(json.dumps({
    "copied_files": len([path for path in OUT_DIR.iterdir() if path.is_file()]),
    "activation_dir": str(OUT_DIR),
    "act_0_shape": list(verified.shape),
}, indent=2))
