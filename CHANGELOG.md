# Changelog

All notable changes to Meso-Regions are documented here.
Versioning follows [SemVer](https://semver.org); 0.x versions may break APIs between minors.

## [0.4.1] — 2026-08-17

### Changed
- Report: per-region colors now only on ECE-positive regions (supervoxels back to a single
  color); each rule-positive curve is drawn in its region's color and annotated with the
  region id, so curves can be matched to the overlays (curve row index == region label)
- Report: per-region layers rendered at higher opacity so hues match the curve colors

## [0.4.0] — 2026-08-17

### Added
- Report: per-region colors for label-map layers (supervoxels, ECE regions) and a new
  `--supervoxels` layer option
- Report: interactive slice slider (`--scroll-plane`, mouse-wheel support, capped at 72
  pre-rendered frames; `--no-scroll` to omit) — still a single self-contained file

## [0.3.0] — 2026-08-17

### Added
- `meso-regions report` — self-contained HTML patient report (`meso_regions/report.py`):
  axial/coronal/sagittal overlays of fluid / pleural-region / ECE masks, per-superspel
  signal-intensity curves classified in-place by the published ECE rule, mask volume table,
  patient-level ECE call, and research-use disclaimer. Single file, no external assets,
  no data upload.
- README: report usage and how to view outputs in 3D Slicer / ITK-SNAP

## [0.2.0] — 2026-08-17

### Added
- `meso-regions` CLI (`meso_regions/cli.py`): `stages`, `example-config`, and
  `run -c config.yaml [--dry-run]` — YAML-driven pipeline execution replacing the
  hard-coded paths in the per-stage scripts
- Console entry point and `pyyaml` dependency in `pyproject.toml`

## [0.1.0] — 2026-08-17

First installable release of the research code behind the ISBI 2026 paper.

### Added
- `pyproject.toml` — the project installs with `pip install -e .`
- README with pipeline overview, install steps, research-use-only disclaimer,
  PREDICT-Meso data access, and citations
- `docs/ROADMAP.md` — packaging/CLI, pleural mask v2, modernization candidates
- `.gitignore`, this changelog

### Changed
- Package renamed `src` → `meso_regions`; all internal imports rewritten

## [pre-0.1.0]

Research code as used for Theodoro et al., IEEE ISBI 2026 (commit `5706ad9` and earlier).
