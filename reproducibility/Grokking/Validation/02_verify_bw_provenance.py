"""Compare a historical BW cache with a fresh distance-only reproduction."""

import argparse
from pathlib import Path

from grokking_velocity_hodge.provenance import compare_bw_files
from grokking_velocity_hodge.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", required=True, type=Path)
    parser.add_argument("--rerun", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_bw_files(args.cached, args.rerun)
    if args.output:
        write_json(args.output, report)
    print(report)
    assert report["conclusion_stable"], "Cached and rerun BW conclusions differ materially"


if __name__ == "__main__":
    main()
