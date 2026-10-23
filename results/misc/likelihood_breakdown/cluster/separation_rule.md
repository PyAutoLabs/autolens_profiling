# Cluster PointSolver: coarsest safe starting scale vs separation (#439)

Production extent (workspace grid ±49.5″), precision 0.001, magnification threshold 0.1; a scale covers a draw-source when no considered oracle root is missed or duplicated. `safe` = largest tested scale with every finer tested scale covering; `≥` = safe at every tested scale, `<` = the finest tested scale misses.

## Draw-sources that miss at some tested scale

| Group | Set | Draw/src | Images | Min img–img ″ | Min img–core ″ (core) | Missing at | Coarsest covering | Coarsest safe |
|---|---|---|---|---|---|---|---|---|
| simple stress 53/73/81 (laptop, 8 scales) | stress | 53/0 | 5 | 0.73 | 0.11 (scaling_4) | 1, 2 | 0.7 | 0.7 |
| simple stress 53/73/81 (laptop, 8 scales) | stress | 73/0 | 7 | 0.76 | 0.30 (scaling_4) | 1, 2 | 0.7 | 0.7 |
| simple stress 53/73/81 (laptop, 8 scales) | stress | 81/1 | 3 | 0.33 | 1.70 (scaling_9) | 0.37, 0.5, 0.7, 1, 2 | 0.4 | 0.3 |
| viz_cluster | stress | 9/0 | 3 | 0.28 | 0.13 (scaling_6) | 0.37, 0.4, 0.7, 1 | 0.5 | 0.3 |
| viz_cluster | prior | 1/0 | 5 | 2.79 | 0.17 (main_1) | 0.2, 0.3, 0.37, 0.4, 0.5, 0.7, 1 | — | < 0.2 |
| viz_cluster | stress | 16/1 | 3 | 0.45 | 0.20 (scaling_0) | 0.3, 0.37, 0.4, 0.5, 0.7, 1 | 0.2 | 0.2 |
| viz_cluster | prior | 4/0 | 5 | 2.75 | 0.22 (main_1) | 0.2, 0.3, 0.37, 0.5, 0.7, 1 | 0.4 | < 0.2 |
| viz_cluster | prior | 2/0 | 5 | 2.70 | 0.32 (main_1) | 0.3, 0.37, 0.4, 0.5, 0.7, 1 | 0.2 | 0.2 |
| viz_cluster | prior | 5/0 | 5 | 2.69 | 0.33 (main_1) | 0.2, 0.3, 0.5, 0.7, 1 | 0.4 | < 0.2 |
| viz_cluster | prior | 6/0 | 5 | 2.72 | 0.33 (main_1) | 0.3, 0.37, 0.4, 0.5, 0.7, 1 | 0.2 | 0.2 |
| viz_cluster | prior | 3/0 | 5 | 2.71 | 0.39 (scaling_3) | 0.3, 0.37, 0.5, 0.7, 1 | 0.4 | 0.2 |
| viz_cluster | prior | 0/0 | 5 | 2.77 | 0.39 (main_1) | 0.3, 0.37, 0.4, 0.5, 0.7, 1 | 0.2 | 0.2 |
| viz_cluster | prior | 8/0 | 5 | 2.76 | 0.42 (main_1) | 0.3, 0.37, 0.4, 0.5, 0.7, 1 | 0.2 | 0.2 |
| viz_cluster | prior | 7/0 | 5 | 2.80 | 0.49 (main_1) | 0.37, 0.5, 0.7, 1 | 0.4 | 0.3 |
| viz_cluster | stress | 11/0 | 5 | 1.89 | 0.70 (main_1) | 0.7, 1 | 0.5 | 0.5 |

Safe at every tested scale: simple (RAL phase 1, 4 scales): 196 draw-sources (≥ 2″); simple stress 53/73/81 (laptop, 8 scales): 3 draw-sources (≥ 2″); viz_cluster: 22 draw-sources (≥ 1″).

## Rule test

`c_required` = the smallest c for which "scale ≤ separation / c is safe" is not contradicted by any observed miss (a miss at scale s contradicts every c ≤ sep / s).

| Unit | Separation | c_required | Binding case | Failures |
|---|---|---|---|---|
| per_root | nearest_image_sep | 13.93 | viz_cluster/1/0@[10.323, 8.227] at 0.2″ | 80 |
| per_root | nearest_core_sep | 11.12 | simple/81/1@[2.321, -2.363] at 0.37″ | 80 |
| per_root | min_feature_sep | 1.66 | viz_cluster/5/0@[10.258, 8.188] at 0.2″ | 80 |
| per_draw_source | min_image_image_sep | 13.93 | viz_cluster/1/0 at 0.2″ | 72 |
| per_draw_source | min_image_core_sep | 4.60 | simple/81/1 at 0.37″ | 72 |
| per_draw_source | min_sep | 1.66 | viz_cluster/5/0 at 0.2″ | 72 |

Claimed rule (scale ≲ separation / 1.06) holds per root: False. Non-monotone draw-sources (a coarser scale covers, a finer one misses): 6.

## A2744 pilot rows (job 412573; row-level only, no per-root record)

| Row | Scale | Prec | Status | Coverage | Prior | Stress | Missed | Max err ″ | RSS MB | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| production_s0.5_p0.01 | 0.5 | 0.01 | diagnostic | 66/69 | 48/49 | 18/20 | 3 | 0.0077 | 8349 | 22077 |
| production_s0.5_p0.001 | 0.5 | 0.001 | diagnostic | 66/69 | 48/49 | 18/20 | 3 | 0.0009 | 12395 | 42011 |
| production_s0.7_p0.01 | 0.7 | 0.01 | diagnostic | 64/69 | 48/49 | 16/20 | 5 | 0.0059 | 13111 | 42011 |
| production_s0.7_p0.001 | 0.7 | 0.001 | diagnostic | 64/69 | 48/49 | 16/20 | 5 | 0.0007 | 16615 | 53082 |
| production_s1_p0.01 | 1.0 | 0.01 | diagnostic | 67/69 | 49/49 | 18/20 | 2 | 0.0077 | 16766 | 53082 |
| control | 1.0 | 0.001 | diagnostic | 67/69 | 49/49 | 18/20 | 2 | 0.0009 | 20122 | 56695 |
| production_s2_p0.01 | 2.0 | 0.01 | diagnostic | 61/69 | 47/49 | 14/20 | 8 | 0.0077 | 20783 | 56695 |
| production_s2_p0.001 | 2.0 | 0.001 | diagnostic | 61/69 | 47/49 | 14/20 | 8 | 0.0009 | 24466 | 67620 |
| bbox_s0.5_p0.01 | 0.5 | 0.01 | diagnostic | 58/69 | 48/49 | 10/20 | 11 | 0.0077 | 23954 | 67620 |
| bbox_s0.5_p0.001 | 0.5 | 0.001 | diagnostic | 58/69 | 48/49 | 10/20 | 11 | 0.0009 | 26618 | 67620 |
| bbox_s0.7_p0.01 | 0.7 | 0.01 | diagnostic | 59/69 | 48/49 | 11/20 | 10 | 0.0059 | 27361 | 67620 |
| bbox_s0.7_p0.001 | 0.7 | 0.001 | diagnostic | 59/69 | 48/49 | 11/20 | 10 | 0.0007 | 30603 | 67620 |
| bbox_s1_p0.01 | 1.0 | 0.01 | diagnostic | 60/69 | 49/49 | 11/20 | 9 | 0.0077 | 30821 | 67620 |
| bbox_s1_p0.001 | 1.0 | 0.001 | diagnostic | 60/69 | 49/49 | 11/20 | 9 | 0.0009 | 33918 | 70739 |
| bbox_s2_p0.01 | 2.0 | 0.01 | diagnostic | 59/69 | 47/49 | 12/20 | 10 | 0.0077 | 34689 | 70739 |
| bbox_s2_p0.001 | 2.0 | 0.001 | diagnostic | 59/69 | 47/49 | 12/20 | 10 | 0.0009 | 38090 | 81528 |
| tight_s1_p0.001 | 1.0 | 0.001 | invalid | — | — | — | — | — | 39412 | — |
| control_mcs40 | 1.0 | 0.001 | invalid | — | — | — | — | — | 39354 | — |
| control_mcs60 | 1.0 | 0.001 | invalid | — | — | — | — | — | 39446 | — |

- `tight_s1_p0.001`: JaxRuntimeError: INTERNAL: Failed to materialize symbols: { (<xla_jit_dylib_23>, { bitcast_concatenate_fusion.1572 }) }
- `control_mcs40`: JaxRuntimeError: INTERNAL: Failed to materialize symbols: { (<xla_jit_dylib_24>, { multiply_power_fusion.1 }) }
- `control_mcs60`: JaxRuntimeError: INTERNAL: Failed to materialize symbols: { (<xla_jit_dylib_26>, { wrapped_scatter.69 }) }
