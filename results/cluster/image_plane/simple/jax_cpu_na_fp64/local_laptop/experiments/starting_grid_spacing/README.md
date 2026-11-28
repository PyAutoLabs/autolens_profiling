# Simple image-plane starting-grid comparison

Paired local CPU runs of the full solved image-plane likelihood at two solver starting grids. Both runs use the same fitted prior-median mass model, solved source centres, 100″ field width and 0.001″ refinement precision, on the profiling head `177c5d10`.

| Starting grid | `point_0` matched | `point_1` matched | Warm median (ms) | Warm p10–p90 (ms) | Absolute JAX–NumPy log L difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0″ · 100 × 100 | 4/5 | 2/2 | 11.778 | 8.840–17.609 | 1.75e-10 |
| 0.25″ · 400 × 400 | 5/5 | 2/2 | 51.101 | 43.292–80.163 | 2.13e-13 |

In this pair the 0.25″ grid took 4.34× as long per evaluation as the 1″ control. Each matched image has a distinct prediction within 0.01″, twice the catalogue positional uncertainty. For the second `point_0` observation the nearest prediction moved from 2.764″ to 0.00745″. The 8.918 ms image-plane headline in `../../runtime.json` came from a separate run, so compare the fine grid with the 11.778 ms control measured alongside it.

Each median uses 200 synchronized JAX FP64 calls after five warm-ups on the same i9-10885H CPU. Recovery was checked only at this one model instance and its two solved source centres. The production solver settings are unchanged.

`grid_1p0.json` and `grid_0p25.json` hold predicted counts, per-observation separations, likelihoods, JIT phases and software revisions. To repeat a measurement, run `python <path>/measure.py 0.25 fine.json` or `python <path>/measure.py 1.0 coarse.json` from the repository root with the CPU environment active.
