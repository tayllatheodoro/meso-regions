# Changelog

All notable changes to Meso-Regions are documented here.
Versioning follows [SemVer](https://semver.org); 0.x versions may break APIs between minors.

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
