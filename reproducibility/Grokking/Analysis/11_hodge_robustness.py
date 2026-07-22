"""Run probe, parameter, and correspondence-null Hodge robustness checks."""

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.robustness import HodgeSweepConfig, run_hodge_robustness
from grokking_velocity_hodge.runtime import configure_grokking_runtime, load_training_meta, write_json


def main() -> None:
    runtime = configure_grokking_runtime()
    experiment = ExperimentConfig.from_environment()
    sweep = HodgeSweepConfig.from_environment()
    epochs = experiment.checkpoint_epochs(load_training_meta(runtime.activation_dir))
    result = run_hodge_robustness(runtime.activation_dir, epochs, experiment, sweep)
    output = runtime.result_dir("grokking_hodge_robustness") / "hodge_robustness.json"
    write_json(output, result)
    print(f"Saved {len(result['records'])} Hodge robustness records: {output}")


if __name__ == "__main__":
    main()
