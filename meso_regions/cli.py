"""Command-line interface for the Meso-Regions pipeline.

Drives :class:`meso_regions.mesoECE.experiment.Experiment` from a YAML config
instead of the per-stage scripts under ``meso_regions/run/``.

Usage:
    meso-regions stages                  # list available pipeline stages
    meso-regions example-config          # print a config template to stdout
    meso-regions run -c config.yaml      # execute the configured pipeline
    meso-regions run -c config.yaml --dry-run   # resolve + print, don't execute
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import meso_regions.run.config as stage_configs

EXAMPLE_CONFIG = """\
# Meso-Regions pipeline configuration
experiment_name: my_experiment

paths:
  images: /path/to/nifti/images          # per-patient, per-time-point NIfTI volumes
  masks: /path/to/pleural_region/masks
  classes: /path/to/classes.csv           # patient id -> class table
  output: /path/to/experiments/output

# Either an explicit list of patient ids...
patients: [44, 62, 63]
# ...or a CSV with an ID column (takes precedence if set):
# patients_csv: /path/to/training_set.csv

threads: 0   # 0 = single-threaded; otherwise number of worker threads

# Stages run in order. `stage` matches a define_config_* / define_* function in
# meso_regions/run/config.py (run `meso-regions stages` for the catalogue);
# remaining keys are passed to it as keyword arguments.
pipeline:
  - stage: full_ece
    ref_t: 270
    filter_size: 3
    with_mask: true
"""


def _stage_catalogue():
    """Map stage name -> config-builder function, introspected from run.config."""
    catalogue = {}
    for name, fn in inspect.getmembers(stage_configs, inspect.isfunction):
        if name.startswith("define_config_"):
            catalogue[name.removeprefix("define_config_")] = fn
        elif name.startswith("define_"):
            catalogue[name.removeprefix("define_")] = fn
    return catalogue


def cmd_stages(_args):
    for name, fn in sorted(_stage_catalogue().items()):
        print(f"{name}{inspect.signature(fn)}")
    return 0


def cmd_example_config(_args):
    print(EXAMPLE_CONFIG, end="")
    return 0


def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _resolve(cfg):
    """Validate the YAML dict and build Experiment kwargs + stage configs."""
    for key in ("experiment_name", "paths", "pipeline"):
        if key not in cfg:
            sys.exit(f"config error: missing required key '{key}'")
    paths = cfg["paths"]
    for key in ("images", "masks", "classes", "output"):
        if key not in paths:
            sys.exit(f"config error: missing required path '{key}'")

    if cfg.get("patients_csv"):
        import pandas as pd
        ids = sorted(pd.read_csv(cfg["patients_csv"])["ID"].tolist())
    else:
        ids = cfg.get("patients")
    if not ids:
        sys.exit("config error: provide 'patients' or 'patients_csv'")

    catalogue = _stage_catalogue()
    stage_dicts = []
    for i, stage in enumerate(cfg["pipeline"]):
        stage = dict(stage)
        name = stage.pop("stage", None)
        if name not in catalogue:
            known = ", ".join(sorted(catalogue))
            sys.exit(f"config error: pipeline[{i}] unknown stage '{name}'. "
                     f"Known stages: {known}")
        try:
            stage_dicts.append(catalogue[name](**stage))
        except TypeError as exc:
            sys.exit(f"config error: pipeline[{i}] stage '{name}': {exc}")

    threads = cfg.get("threads", 0)
    if threads:
        threads = min(threads, os.cpu_count() or 1, len(ids))

    exp_kwargs = dict(
        experiment_name=cfg["experiment_name"],
        path_images=Path(paths["images"]),
        path_masks=Path(paths["masks"]),
        path_classes=Path(paths["classes"]),
        path_experiments=Path(paths["output"]),
        ids=ids,
        config=stage_dicts,
        threads=threads,
    )
    return exp_kwargs


def cmd_run(args):
    cfg = _load_yaml(args.config)
    exp_kwargs = _resolve(cfg)

    if args.dry_run:
        printable = {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in exp_kwargs.items()}
        print(json.dumps(printable, indent=2))
        print("\ndry run: configuration is valid; nothing executed.")
        return 0

    from meso_regions.mesoECE.experiment import Experiment
    experiment = Experiment(**exp_kwargs)
    experiment.execute_pipeline()
    metrics = experiment.classifier_metrics()
    print(metrics)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="meso-regions",
        description="Automated 4D ECE biomarker segmentation in pleural "
                    "DCE-MRI. Research use only — not a medical device.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="list available pipeline stages")
    sub.add_parser("example-config", help="print a YAML config template")

    run_p = sub.add_parser("run", help="run a pipeline from a YAML config")
    run_p.add_argument("-c", "--config", required=True,
                       help="path to YAML configuration file")
    run_p.add_argument("--dry-run", action="store_true",
                       help="validate and print the resolved configuration "
                            "without executing")

    args = parser.parse_args(argv)
    return {"stages": cmd_stages,
            "example-config": cmd_example_config,
            "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
