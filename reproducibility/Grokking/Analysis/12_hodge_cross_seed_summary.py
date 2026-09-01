"""Regenerate eight-seed Hodge statistics and paper tables from saved results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from grokking_velocity_hodge.reporting import (
    build_cross_seed_hodge_summary,
    write_cross_seed_hodge_outputs,
)
from grokking_velocity_hodge.seed_sweep import load_seed_sweep_config

SEEDS = (598, 599, 777, 1001, 2027, 3141, 4096, 8080)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundled_manifest(artifact_dir: Path) -> int:
    manifest_path = artifact_dir / "remote_artifact_manifest_8seed.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        remote_path = str(entry["path"])
        remote_name = Path(remote_path).name
        if remote_name == "hodge_robustness.json":
            seed = re.search(r"/seed(\d+)/", remote_path)
            if seed is None:
                raise ValueError(f"Cannot identify Hodge seed in manifest path: {remote_path}")
            local_name = f"hodge_robustness_seed{seed.group(1)}.json"
        elif remote_name == "training.json":
            seed = re.search(r"grokking_seed(\d+)_hodge", remote_path)
            local_name = f"training_seed{seed.group(1) if seed else 598}.json"
        else:
            local_name = remote_name
        local_path = artifact_dir / local_name
        if not local_path.exists():
            raise FileNotFoundError(f"Manifest artifact is missing: {local_path}")
        if local_path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Artifact size does not match manifest: {local_path}")
        if sha256(local_path) != entry["sha256"]:
            raise ValueError(f"Artifact SHA-256 does not match manifest: {local_path}")
    return len(entries)


def bundled_inputs(artifact_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "training_seed": seed,
            "hodge_path": artifact_dir / f"hodge_robustness_seed{seed}.json",
            "training_path": artifact_dir / f"training_seed{seed}.json",
        }
        for seed in SEEDS
    ]


def pipeline_inputs(config_path: Path) -> list[dict[str, object]]:
    config = load_seed_sweep_config(config_path)
    return [
        {
            "training_seed": int(run["data_seed"]),
            "hodge_path": Path(run["output_root"])
            / "results"
            / "grokking_hodge_robustness"
            / "hodge_robustness.json",
            "training_path": Path(run["activation_dir"]) / "training.json",
        }
        for run in config["runs"]
    ]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=root / "reproducibility" / "artifacts" / "eight_seed_hodge",
        help="Directory containing bundled per-seed Hodge and training JSON files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Instead read fresh per-seed outputs at paths in this seed-sweep config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Machine-readable output directory (defaults below the artifact directory).",
    )
    parser.add_argument(
        "--paper-generated-dir",
        type=Path,
        default=root / "generated",
        help="Directory for generated LaTeX macros and table rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        inputs = pipeline_inputs(args.config)
    else:
        verified = verify_bundled_manifest(args.artifact_dir)
        print(f"Verified {verified} bundled source artifacts against the saved SHA-256 manifest.")
        inputs = bundled_inputs(args.artifact_dir)
    output_dir = args.output_dir or args.artifact_dir / "generated"
    summary = build_cross_seed_hodge_summary(inputs)
    written = write_cross_seed_hodge_outputs(summary, output_dir, args.paper_generated_dir)

    fixed = summary["phase_conventions"]["fixed"]
    event = summary["phase_conventions"]["event_aligned"]
    for label, result in (("fixed", fixed), ("event-aligned", event)):
        estimate = result["seed_level_inference"][
            "baseline_post_minus_transition_coexact_minus_exact"
        ]
        direction = result["direction_counts"]["seed_setting_shift_toward_exact"]
        print(
            f"{label}: mean={estimate['mean']:.6f}, "
            f"95% CI=[{estimate['interval'][0]:.6f}, {estimate['interval'][1]:.6f}], "
            f"direction={direction['count']}/{direction['total']}"
        )
    print("Generated outputs:")
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
