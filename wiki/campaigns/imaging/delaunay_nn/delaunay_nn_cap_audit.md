# DelaunayNN fixed-shape cap audit

## Decision

Keep both `DelaunayNN` static limits at **32**. Cap 16 is demonstrably too low.
Cap 24 overflowed in the local audit but sat exactly on the platform-sensitive
boundary in GitHub CI, so it does not provide a robust safety margin. Cap 64
roughly doubles the cap-32 geometry cost without being required by the audit.

## Correctness audit

The workspace assertion
`scripts/misc/jax_assertions/delaunay_nn_caps.py` constructs a 1,200-vertex
Hilbert image mesh from an arc-like adapt image and ray-traces both that mesh
and 4,421 image pixels through 101 Isothermal + external-shear mass models.
The ensemble contains a 25-step local mass-model trajectory, 75 broad prior
draws, and a fixed stress geometry. It audits ordinary mapper queries and all
4,800 split-cross points used by split regularization.

The untruncated cap-64 reference observed:

| Quantity | Local maximum | Rows above 16 | Rows above 24 |
|---|---:|---:|---:|
| Main natural neighbours | 27 | 198 | 8 |
| Main cavity triangles | 25 | 107 | 2 |
| Split natural neighbours | 21 | 28 | 0 |
| Split cavity triangles | 19 | 12 | 0 |

The main-neighbour 99th, 99.9th and 99.99th percentiles were 9, 14 and 22;
the corresponding split-neighbour percentiles were 9, 11 and 15. Thus 16 is
usually sufficient but fails in the tail—the precise failure mode that would
otherwise create rare NaN likelihood samples as the mass model changes.

The worst geometry was rerun explicitly at caps 16, 24 and 32. Locally, caps
16 and 24 reported 94 and 4 overflow rows respectively; 32 completed without
main or split overflow. On the GitHub Actions runners the cap-24 rerun produced
zero overflow rows, while cap 16 still overflowed. This small qhull/platform
difference makes 24 a boundary result rather than a dependable production cap;
32 covered both environments.

## Runtime trade-off

The versioned local CPU benchmark uses 1,200 vertices, 15,000 data queries and
the full mapper path, including split regularization:

| Mapper | Warm median | Relative to Delaunay |
|---|---:|---:|
| Barycentric Delaunay | 0.0667 s | 1.00× |
| DelaunayNN cap 16 | 0.1923 s | 2.88× |
| DelaunayNN cap 24 | 0.2400 s | 3.59× |
| DelaunayNN cap 32 | 0.3177 s | 4.76× |
| DelaunayNN cap 64 | 0.5624 s | 8.43× |

The corresponding laptop-GPU run used the same 1,200 vertices, 15,000 queries
and query chunk of 256 on an NVIDIA GeForce RTX 2060 Max-Q. It recorded ten
warm calls to span the device's dynamic clock states:

| Mapper | CPU warm median | GPU warm median | GPU speed-up | Relative to GPU Delaunay |
|---|---:|---:|---:|---:|
| Barycentric Delaunay | 0.0667 s | 0.0440 s | 1.52× | 1.00× |
| DelaunayNN cap 16 | 0.1923 s | 0.1194 s | 1.61× | 2.71× |
| DelaunayNN cap 24 | 0.2400 s | 0.1966 s | 1.22× | 4.46× |
| DelaunayNN cap 32 | 0.3177 s | 0.2396 s | 1.33× | 5.44× |
| DelaunayNN cap 64 | 0.5624 s | 0.4422 s | 1.27× | 10.04× |

On the laptop GPU the production cap-32 mapper is faster in absolute terms
(0.2396 s rather than 0.3177 s), but ordinary Delaunay appeared to benefit
more, the overhead reading 5.44× against 4.76× on CPU. **The A100 run below
shows that reading was an RTX 2060 artifact — do not carry it forward.**

## A100 (SLURM job 334949, 2026-08-13)

Same configuration again — 1,200 vertices, 15,000 queries, query chunk 256, ten
warm repeats — on an NVIDIA A100 80GB PCIe on the RAL `gpu` partition, at the
same source revisions as the CPU and laptop columns (PyAutoArray `5dedb5e9`,
PyAutoLens `13a4655c`; the job log records all five).

| Mapper | CPU | RTX 2060 | **A100** | A100 vs CPU | A100 vs 2060 |
|---|---:|---:|---:|---:|---:|
| Barycentric Delaunay | 0.0667 s | 0.0440 s | **0.0366 s** | 1.82× | 1.20× |
| DelaunayNN cap 16 | 0.1923 s | 0.1194 s | **0.0975 s** | 1.97× | 1.22× |
| DelaunayNN cap 24 | 0.2400 s | 0.1966 s | **0.1283 s** | 1.87× | 1.53× |
| DelaunayNN cap 32 | 0.3177 s | 0.2396 s | **0.1573 s** | 2.02× | 1.52× |
| DelaunayNN cap 64 | 0.5624 s | 0.4422 s | **0.2677 s** | 2.10× | 1.65× |

Overhead relative to ordinary Delaunay on each device:

| Mapper | CPU | RTX 2060 | **A100** |
|---|---:|---:|---:|
| DelaunayNN cap 16 | 2.88× | 2.71× | **2.66×** |
| DelaunayNN cap 24 | 3.59× | 4.46× | **3.51×** |
| DelaunayNN cap 32 | 4.76× | 5.44× | **4.30×** |
| DelaunayNN cap 64 | 8.43× | 10.04× | **7.32×** |

**The DelaunayNN overhead does not grow on GPU — it shrinks.** On the A100 the
production cap-32 mapper costs 4.30× ordinary Delaunay, *lower* than the 4.76×
measured on CPU, and well under the laptop's 5.44×. Every cap improves on its
CPU ratio.

The laptop reading inverted because its Delaunay baseline was unstable, not
because the GPU disadvantages DelaunayNN. Across the ten warm calls the
RTX 2060's baseline spanned a factor of **1.58** (0.0363--0.0573 s, dynamic
clock states on a Max-Q part) while the A100's spanned **1.007**
(0.03651--0.03675 s). A noisy denominator inflated the ratio; this is the same
instability that moved the five-repeat laptop ratio from 4.93× to 5.92×. The
A100 numbers are the ones to quote for HPC throughput.

Compilation does not follow the warm trend and is not accelerated by the
better card: compile plus first call on the A100 was 6.604 s for Delaunay and
3.650 s for cap-32 DelaunayNN, against 0.788 s / 3.900 s on the laptop GPU and
0.351 s / 1.242 s on CPU. The A100's slower cold Delaunay compile is
first-call XLA autotuning on the node, a one-off; the warm timings above remain
the relevant per-likelihood geometry cost.

Capacity was identical on all three devices — max cavity 11 and 13 neighbours
(main), 9 and 11 (split), with every overflow and degeneracy count zero — so
the cap-32 static shape carries the same margin on HPC as locally.

The cap-32 choice remains the smallest tested static shape with a robust margin
across the local and CI audits, and the A100 profile this note previously said
was owed is now done.
