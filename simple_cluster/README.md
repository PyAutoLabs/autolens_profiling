# Simple cluster reference

The local simulator and both cluster likelihoods use the seven observed positions and the velocity-dispersion dPIE tracer from the [reviewed gallery fixture](https://github.com/PyAutoLabs/autolens_visualization/tree/017c8673aba5a579b2cba3318a3d1bae34b9b595/dataset/cluster/simple).

`tracer.json` is the gallery file. `point_datasets.csv` was exported with `al.output_to_csv` from the two published point JSONs in family order. It preserves the five and two image coordinates, their 0.005″ uncertainties and source redshifts. The member CSV records the ten scaling galaxies’ centres and luminosities from the tracer. The simulator still measures the forward solve independently. It copies the reference observations into the fit dataset.

The old six-image local CPU rows are retained in [the legacy b0 archive](../archive/2026/11/simple_b0_six_images). Earlier measurements named Simple may use that model. Use the recorded gallery revision and family counts when comparing runtimes.
