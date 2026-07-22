"""Measure analysis stability across fixed probe subsets."""

from pathlib import Path

import numpy as np

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.controls import aggregate_probe_subsets, analyse_probe_subset
from grokking_velocity_hodge.runtime import configure_grokking_runtime, load_training_meta, write_json


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ACT_DIR = GROKKING.activation_dir
    OUT_DIR = GROKKING.result_dir("grokking_probe_subset_robustness")

    SUBSET_SIZE = CONFIG.probe_subset_size
    SUBSET_SEEDS = list(CONFIG.probe_subset_seeds)
    PCA_DIM = CONFIG.pca_dim
    KNN = CONFIG.knn
    K_SPEC = CONFIG.spectral_components
    EPS = CONFIG.resolvent_epsilon
    TAU_VALUES = list(CONFIG.heat_scales)

    print(f"ACT_DIR={ACT_DIR}")
    EPOCHS = CONFIG.checkpoint_epochs(load_training_meta(ACT_DIR))

    x0 = np.load(ACT_DIR / "act_0.npy")
    n_total = x0.shape[0]
    del x0
    print(f"n_total={n_total}, subset_size={SUBSET_SIZE}")

    subsets = []
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        subset_idx = np.sort(rng.choice(n_total, size=SUBSET_SIZE, replace=False))
        subsets.append(
            analyse_probe_subset(
                ACT_DIR,
                seed,
                subset_idx,
                epochs=EPOCHS,
                tau_values=TAU_VALUES,
                pca_dim=PCA_DIM,
                knn=KNN,
                k_spec=K_SPEC,
                eps=EPS,
            )
        )

    aggregate = aggregate_probe_subsets(subsets, TAU_VALUES)

    payload = {
        "config": {
            "subset_size": SUBSET_SIZE,
            "subset_seeds": SUBSET_SEEDS,
            "epochs": EPOCHS,
            "pca_dim": PCA_DIM,
            "knn": KNN,
            "k_spec": K_SPEC,
            "eps": EPS,
            "tau_values": TAU_VALUES,
            "activation_dir": str(ACT_DIR),
        },
        "subsets": subsets,
        "aggregate": aggregate,
    }

    write_json(OUT_DIR / "09_probe_subset_robustness.json", payload)
    write_json(OUT_DIR / "probe_subset_robustness.json", payload)

    print(f"Aggregate: {aggregate}")
    print("Saved 09 probe-subset robustness.")

    qa_out_dir = Path(OUT_DIR)
    print("json exists", (qa_out_dir / "09_probe_subset_robustness.json").exists())

    print("subsets", len(subsets), "expected", len(SUBSET_SEEDS))
    print("subset seeds", [row["seed"] for row in subsets])

    print("expected tau", TAU_VALUES)
    print("heat tau keys", sorted(aggregate["heat_kernel"].keys()))

    print("collapse ratio mean", aggregate["effective_dimension"]["ratio_final_over_0"]["mean"])
    collapse_ratio = aggregate["effective_dimension"]["ratio_final_over_0"]
    print("collapse ratio sd", collapse_ratio.get("sd", collapse_ratio.get("std")))

    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            a = set(subsets[i]["subset_idx"])
            b = set(subsets[j]["subset_idx"])
            print(subsets[i]["seed"], subsets[j]["seed"], "overlap", len(a & b))


if __name__ == "__main__":
    main()
