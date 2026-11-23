# External solar-cycle indices

This directory stores public physical indices used only after the PR24
reference baseline. Raw source files are retained separately from processed
tables.

## Layout

- `raw/f107/`: NCEI Penticton/Ottawa nonontime flux files and one SWPC JSON
  snapshot used to extend the monthly value through 2026-10.
- `raw/wso_polar/`: WSO `Polar.html` snapshot.
- `processed/`: monthly/10-day tables with explicit missing and coverage fields.

## Rules

1. Missing markers are converted to NaN; they are never treated as zero.
2. Short gaps may be linearly interpolated for at most two months; longer gaps
   remain NaN and must be represented by a mask in model inputs.
3. Integrated physical indices use their own scaler; they never share scaling
   parameters with monthly SSN.
4. F10.7 observed flux is retained separately from adjusted/absolute values.
5. WSO polar field keeps north, south, and averaged values separately.
6. Data acquisition and processing are reproducible via
   `scripts/build_external_indices.py`.

## Sources

- F10.7: NOAA NCEI, Penticton/Ottawa nonontime flux archive.
- Recent F10.7 monthly extension: NOAA SWPC observed solar-cycle indices JSON.
- Polar field: Wilcox Solar Observatory, Stanford University.

The local SSN source and license are documented under `../SILSO/README.md`.
