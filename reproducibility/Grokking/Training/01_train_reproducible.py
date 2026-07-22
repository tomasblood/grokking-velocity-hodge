"""Train modular addition and save reproducible activation checkpoints."""

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.runtime import configure_grokking_runtime, ensure_dir, notebook_param, write_json


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()

    P = CONFIG.modulus
    D_MODEL = CONFIG.d_model
    N_EPOCHS = CONFIG.max_epoch
    DATA_SEED = CONFIG.data_seed
    N_SUB = CONFIG.probe_size
    SAVE_EVERY = CONFIG.save_every
    TRAIN_FRAC = CONFIG.train_fraction
    PROBE_SEED = CONFIG.probe_seed
    FORCE = notebook_param("GROKKING_FORCE", "0").strip().lower() in {"1", "true", "yes"}

    default_tmp = Path(tempfile.gettempdir()) / "grokking_acts_v6"
    TMP_DIR = ensure_dir(Path(notebook_param("GROKKING_TMP_DIR", str(default_tmp))))
    OUT_DIR = ensure_dir(Path(notebook_param("GROKKING_ACTIVATION_DIR", str(GROKKING.activation_dir))))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    existing_outputs = list(OUT_DIR.glob("act_*.npy"))
    assert not (existing_outputs and not FORCE), f"{OUT_DIR} already contains activation snapshots"
    if FORCE:
        for directory in (OUT_DIR, TMP_DIR):
            generated = [
                *directory.glob("act_*.npy"),
                directory / "gt_labels.npy",
                directory / "training.json",
            ]
            for path in generated:
                if path.is_file():
                    path.unlink()

    print(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )

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
                val_acc = (
                    (model(val_data[:, :2])[:, -1, :].argmax(-1) == val_data[:, 2]).float().mean().item()
                )
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
                print(
                    f"{epoch}: loss={train_loss:.4f}; val_acc={val_acc:.4f}; elapsed={time.time() - t0:.0f}s"
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training finished in {time.time() - t0:.0f}s; final val_acc={val_accs[-1]:.4f}")

    sample = np.load(TMP_DIR / "act_0.npy")
    expected_shape = (N_SUB, D_MODEL * 4)
    assert sample.shape == expected_shape, f"Expected {expected_shape}; got {sample.shape}"

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
            "model_seed": f"torch.manual_seed({DATA_SEED}); no HookedTransformerConfig seed",
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

    generated_outputs = [
        *sorted(TMP_DIR.glob("act_*.npy")),
        TMP_DIR / "gt_labels.npy",
        TMP_DIR / "training.json",
    ]
    for path in generated_outputs:
        assert path.is_file(), f"Missing generated training output: {path}"
        shutil.copy2(path, OUT_DIR / path.name)

    verified = np.load(OUT_DIR / "act_0.npy")
    print(
        json.dumps(
            {
                "copied_files": len([path for path in OUT_DIR.iterdir() if path.is_file()]),
                "activation_dir": str(OUT_DIR),
                "act_0_shape": list(verified.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
