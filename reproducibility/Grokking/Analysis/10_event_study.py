"""Construct the seed-aligned grokking event study."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.runtime import (
    ACCENT_COLOR,
    GROKKING_ALPHA,
    GROKKING_SHADE,
    MAIN_COLOR,
    SECONDARY_COLOR,
    TEXT_COLOR,
    configure_grokking_runtime,
    notebook_param,
    set_paper_style,
    write_json,
)
from grokking_velocity_hodge.seed_sweep import load_seed_sweep_config


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ROOT = GROKKING.root
    FIG_DIR = GROKKING.figure_dir
    OUT_DIR = GROKKING.result_dir("grokking_event_study")
    set_paper_style()
    print(f"ROOT: {ROOT}")

    DEFAULT_SUMMARY_JSON = ""

    EVENT_SUMMARY_JSON = notebook_param("GROKKING_EVENT_SUMMARY_JSON", DEFAULT_SUMMARY_JSON).strip()
    EVENT_RUNS_JSON = notebook_param("GROKKING_EVENT_RUNS_JSON", "").strip()

    TRANSITION_WINDOW_DELTA = CONFIG.event_transition_delta
    EVENT_X_LIMITS = CONFIG.event_x_limits
    EVENT_GRID_STEP = CONFIG.event_grid_step
    THRESHOLD_FOR_ALIGNMENT = CONFIG.event_alignment_threshold

    def _default_event_runs() -> list[dict]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "seed_sweep.json"
        config = load_seed_sweep_config(config_path)
        colors = [MAIN_COLOR, ACCENT_COLOR, SECONDARY_COLOR]
        return [
            {
                "seed": str(run["data_seed"]),
                "key": run["key"],
                "label": run["label"].lower(),
                "root": run["output_root"],
                "activation_dir": run["activation_dir"],
                "color": colors[index % len(colors)],
            }
            for index, run in enumerate(config["runs"])
        ]

    RUNS = json.loads(EVENT_RUNS_JSON) if EVENT_RUNS_JSON else _default_event_runs()
    assert RUNS, "At least one event-study run is required"
    print("Event-study runs:")
    for run in RUNS:
        print(f"  {run['seed']}: {run['root']}")

    def resolve_path(path: str | Path, base: Path | None = None) -> Path:
        path_text = str(path)
        if path_text.startswith("dbfs:/"):
            return Path("/dbfs") / path_text[len("dbfs:/") :].lstrip("/")
        path = Path(path)
        if path.is_absolute():
            return path
        return (base or ROOT) / path

    def load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    def load_optional_json(path_text: str) -> dict:
        if not path_text:
            return {}
        path = resolve_path(path_text)
        if not path.exists():
            print(f"Optional summary not found: {path}")
            return {}
        return load_json(path)

    ROBUSTNESS = load_optional_json(EVENT_SUMMARY_JSON)
    ROBUSTNESS_RUNS = ROBUSTNESS.get("run_summaries", {})

    def run_root(run: dict) -> Path:
        return resolve_path(run["root"])

    def result_root(run: dict) -> Path:
        if run.get("result_root"):
            return resolve_path(run["result_root"], run_root(run))
        return run_root(run) / "results"

    def activation_dir(run: dict) -> Path:
        if run.get("activation_dir"):
            return resolve_path(run["activation_dir"], run_root(run))
        candidates = [
            run_root(run) / "results" / "grokking_acts_v6",
            run_root(run) / "activations",
            run_root(run) / "results" / "grokking_acts",
        ]
        if str(run.get("seed")) == "598":
            candidates.insert(0, GROKKING.activation_dir)
        for candidate in candidates:
            if (candidate / "training.json").exists():
                return candidate
        return candidates[0]

    def training_series(run: dict) -> tuple[np.ndarray, np.ndarray]:
        training = load_json(activation_dir(run) / "training.json")
        val_accs = training.get("val_accs", training.get("test_accs", []))
        epochs = training.get("epochs", training.get("saved_epochs", list(range(len(val_accs)))))
        return np.asarray(epochs, dtype=float), np.asarray(val_accs, dtype=float)

    def first_threshold_epoch(run: dict, threshold: float = THRESHOLD_FOR_ALIGNMENT) -> float:
        if "t_grok" in run:
            return float(run["t_grok"])
        summary = ROBUSTNESS_RUNS.get(run.get("key", ""), {})
        summary_key = f"first_epoch_val_ge_{str(threshold).replace('.', '_')}"
        if summary_key in summary:
            return float(summary[summary_key])
        epochs, accs = training_series(run)
        above = epochs[accs >= threshold]
        assert len(above), f"No validation accuracy >= {threshold} for run {run['seed']}"
        return float(above[0])

    def record_series(
        records: list[dict], x_key: str, y_key: str, t_grok: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = sorted(records, key=lambda row: float(row[x_key]))
        x = np.array([float(row[x_key]) - t_grok for row in rows], dtype=float)
        y = np.array([float(row[y_key]) for row in rows], dtype=float)
        return x, y

    def mapping_series(mapping: dict[str, float], t_grok: float) -> tuple[np.ndarray, np.ndarray]:
        pairs = sorted((float(k), float(v)) for k, v in mapping.items())
        x = np.array([item[0] - t_grok for item in pairs], dtype=float)
        y = np.array([item[1] for item in pairs], dtype=float)
        return x, y

    def validation_series(run: dict, t_grok: float) -> tuple[np.ndarray, np.ndarray]:
        summary = ROBUSTNESS_RUNS.get(run.get("key", ""), {})
        records = summary.get("circular_fourier", {}).get("series")
        if records:
            return record_series(records, "epoch", "val_acc", t_grok)
        epochs, accs = training_series(run)
        return epochs - t_grok, accs

    def participation_ratio_from_activation(path: Path) -> float:
        x = np.load(path).astype(np.float64)
        x = x.reshape(x.shape[0], -1)
        x = x - x.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        eigvals = (singular_values**2) / max(1, x.shape[0] - 1)
        total = float(np.sum(eigvals))
        return float((total**2) / max(float(np.sum(eigvals**2)), 1e-30))

    def effective_dimension_series(run: dict, t_grok: float) -> tuple[np.ndarray, np.ndarray]:
        summary = ROBUSTNESS_RUNS.get(run.get("key", ""), {})
        series = summary.get("effective_dimension", {}).get("series")
        if series:
            return mapping_series(series, t_grok)

        epochs, _ = training_series(run)
        act_dir = activation_dir(run)
        values = []
        valid_epochs = []
        for ep in epochs.astype(int):
            path = act_dir / f"act_{ep}.npy"
            if path.exists():
                valid_epochs.append(float(ep))
                values.append(participation_ratio_from_activation(path))
        assert values, f"No activation files found for effective dimension in {act_dir}"
        return np.asarray(valid_epochs, dtype=float) - t_grok, np.asarray(values, dtype=float)

    def bw_series(run: dict, t_grok: float) -> tuple[np.ndarray, np.ndarray]:
        root = result_root(run)
        current = root / "grokking_resolvent_bw" / "resolvent_bw_results.json"
        legacy = root / "grokking_bw_geodesic" / "bw_geodesic_results.json"
        data = load_json(current if current.exists() else legacy)
        series = data["bw_distances_consecutive"]
        x = np.asarray(series["midpoint_epochs"], dtype=float) - t_grok
        y = np.asarray(series["distances"], dtype=float)
        return x, y

    def heat_series(run: dict, t_grok: float, tau: str = "1.0") -> tuple[np.ndarray, np.ndarray]:
        data = load_json(result_root(run) / "grokking_heat_kernel" / "heat_kernel_bw_results.json")
        series = data["bw_series"][tau]
        x = np.asarray(series["midpoint_epochs"], dtype=float) - t_grok
        y = np.asarray(series["distances"], dtype=float)
        return x, y

    def hodge_series(run: dict, t_grok: float, smooth_window: int = 3) -> tuple[np.ndarray, np.ndarray]:
        data = load_json(result_root(run) / "grokking_dg_velocity_hodge" / "velocity_hodge.json")
        x, y = record_series(data["pairs"], "midpoint", "coexact", t_grok)
        if smooth_window <= 1 or len(y) < smooth_window:
            return x, y
        pad = smooth_window // 2
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        smoothed = np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
        return x, smoothed

    def interpolate_stack(
        series_by_seed: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray
    ) -> np.ndarray:
        rows = []
        for x, y in series_by_seed.values():
            order = np.argsort(x)
            xs = x[order]
            ys = y[order]
            row = np.interp(grid, xs, ys)
            row[(grid < xs.min()) | (grid > xs.max())] = np.nan
            rows.append(row)
        return np.vstack(rows)

    panels = [
        {
            "letter": "A",
            "name": "validation",
            "title": "Validation accuracy",
            "ylabel": "accuracy",
            "ylim": (-0.03, 1.04),
            "series": {},
        },
        {
            "letter": "B",
            "name": "effective_dimension",
            "title": "Effective dimension",
            "ylabel": "participation",
            "ylim": None,
            "series": {},
        },
        {
            "letter": "C",
            "name": "resolvent_bw",
            "title": "Resolvent BW",
            "ylabel": r"$d_{BW}$",
            "ylim": None,
            "series": {},
        },
        {
            "letter": "D",
            "name": "heat_tau_1",
            "title": r"Heat-kernel BW, $\tau=1$",
            "ylabel": r"$d_{BW}$",
            "ylim": None,
            "series": {},
        },
        {
            "letter": "E",
            "name": "hodge_coexact",
            "title": "Hodge coexact fraction",
            "ylabel": "fraction",
            "ylim": (0.0, 0.82),
            "series": {},
        },
    ]
    panel_by_name = {panel["name"]: panel for panel in panels}
    run_metadata = []

    for run in RUNS:
        seed = str(run["seed"])
        t_grok = first_threshold_epoch(run)
        run_metadata.append(
            {
                "seed": seed,
                "key": run.get("key"),
                "label": run.get("label", seed),
                "t_grok": t_grok,
                "root": str(run_root(run)),
                "result_root": str(result_root(run)),
                "activation_dir": str(activation_dir(run)),
            }
        )
        panel_by_name["validation"]["series"][seed] = validation_series(run, t_grok)
        panel_by_name["effective_dimension"]["series"][seed] = effective_dimension_series(run, t_grok)
        panel_by_name["resolvent_bw"]["series"][seed] = bw_series(run, t_grok)
        panel_by_name["heat_tau_1"]["series"][seed] = heat_series(run, t_grok, tau="1.0")
        panel_by_name["hodge_coexact"]["series"][seed] = hodge_series(run, t_grok)
        print(f"Loaded event-study series for seed {seed}; t_grok={t_grok:.0f}")

    def add_panel_label(ax, label: str) -> None:
        ax.text(
            -0.14,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            ha="left",
            color=TEXT_COLOR,
        )

    def plot_panel(ax, panel: dict) -> None:
        grid = np.arange(EVENT_X_LIMITS[0], EVENT_X_LIMITS[1] + EVENT_GRID_STEP, EVENT_GRID_STEP)
        stack = interpolate_stack(panel["series"], grid)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
        valid = np.isfinite(mean)

        ax.axvspan(
            TRANSITION_WINDOW_DELTA[0],
            TRANSITION_WINDOW_DELTA[1],
            color=GROKKING_SHADE,
            alpha=GROKKING_ALPHA,
            lw=0,
            zorder=0,
        )
        ax.axvline(0, color="#495057", linestyle="--", linewidth=0.9, alpha=0.85, zorder=1)
        ax.fill_between(
            grid[valid], mean[valid] - sd[valid], mean[valid] + sd[valid], color="#9AA0A6", alpha=0.15, lw=0
        )

        for run in RUNS:
            seed = str(run["seed"])
            x, y = panel["series"][seed]
            ax.plot(x, y, color=run["color"], lw=1.35, alpha=0.85, label=run.get("label", seed))

        ax.plot(grid[valid], mean[valid], color=TEXT_COLOR, lw=2.0, label="mean")
        ax.set_title(panel["title"], pad=2.5)
        ax.set_ylabel(panel["ylabel"])
        ax.set_xlim(*EVENT_X_LIMITS)
        if panel.get("ylim") is not None:
            ax.set_ylim(*panel["ylim"])
        add_panel_label(ax, panel["letter"])

    fig = plt.figure(figsize=(7.15, 5.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.66, wspace=0.34)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]

    for ax, panel in zip(axes, panels):
        plot_panel(ax, panel)
        ax.set_xlabel(r"$\Delta$ epoch")

    axes[0].text(0.52, 0.13, "val >= 0.5", transform=axes[0].transAxes, fontsize=7.2, color="#495057")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.8,
        columnspacing=1.4,
    )

    fig.savefig(FIG_DIR / "fig_grokking_event_study.pdf")
    fig.savefig(FIG_DIR / "fig_grokking_event_study.png")
    plt.close(fig)

    print(f"Saved: {FIG_DIR / 'fig_grokking_event_study.pdf'}")

    serialised_panels = {}
    for panel in panels:
        serialised_panels[panel["name"]] = {
            seed: {
                "delta_epoch": [float(v) for v in series[0]],
                "value": [float(v) for v in series[1]],
            }
            for seed, series in panel["series"].items()
        }

    payload = {
        "config": {
            "alignment": f"epoch minus first validation accuracy >= {THRESHOLD_FOR_ALIGNMENT}",
            "transition_window_delta_epoch": list(TRANSITION_WINDOW_DELTA),
            "x_limits": list(EVENT_X_LIMITS),
            "event_summary_json": EVENT_SUMMARY_JSON,
        },
        "runs": run_metadata,
        "panels": serialised_panels,
        "outputs": {
            "pdf": str(FIG_DIR / "fig_grokking_event_study.pdf"),
            "png": str(FIG_DIR / "fig_grokking_event_study.png"),
        },
    }

    out_json = OUT_DIR / "event_study_results.json"
    write_json(out_json, payload)
    print(f"Saved: {out_json}")

    print("json exists", out_json.exists())
    print("event-study figure exists", (FIG_DIR / "fig_grokking_event_study.pdf").exists())

    print("runs", len(run_metadata), "expected", len(RUNS))
    print("seeds", [row["seed"] for row in run_metadata])

    for row in run_metadata:
        print(row["seed"], "t_grok", row["t_grok"])

    for panel in panels:
        counts = {seed: len(series[0]) for seed, series in panel["series"].items()}
        print(panel["name"], counts)


if __name__ == "__main__":
    main()
