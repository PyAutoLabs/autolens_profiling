"""Current imaging/MGE runtime construction, with no work performed on import.

Setup remains outside the JIT boundary. Callers own dataset acquisition, timing,
precision selection and output; this module never simulates or writes files.
"""

MASK_RADIUS = 3.5


def prepare_dataset(dataset, *, mask_radius=MASK_RADIUS, use_sparse_operator=False):
    import autolens as al

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    if use_sparse_operator:
        # The pure-MGE-source cell uses LinearObjFuncList for every linear
        # obj, so the inversion factory's all-LinearObjFuncList short-circuit
        # fires and ``InversionImagingMapping`` is still chosen even after
        # ``apply_sparse_operator`` attaches a sparse_operator to the dataset.
        # We still call ``apply_sparse_operator`` for parity with the pix /
        # Delaunay cells so the harness-level cost (kernel construction,
        # serialisation) is included in the timing — but the per-eval cost
        # remains dense. The JSON's ``inversion_path`` records what the flag
        # asked for; downstream synthesis cross-references with the
        # ``InversionImagingSparse``-vs-``InversionImagingMapping`` factory
        # decision when interpreting the MGE row.
        dataset = dataset.apply_sparse_operator()

    return dataset


def build_model(*, mask_radius=MASK_RADIUS):
    import autofit as af
    import autolens as al

    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    # Mass and shear fixed to simulator truth (not GaussianPrior) because
    # tracing GaussianPrior-backed mass params through this script's
    # ``mapping_matrix_from_params`` JIT trigger a pre-existing xp=np/jnp
    # propagation bug in autogalaxy/profiles/mass/total/isothermal.py:108
    # (Isothermal.deflections_yx_2d_from called with xp=np on traced inputs).
    # The bug is specific to this script's MGE-lens-light + over-sampled-LP
    # combination; the likelihood-only imaging/mge_gradients.py uses the
    # same pattern without the failing JIT and works under Option A.
    mass = af.Model(al.mp.Isothermal)
    mass.centre = (0.0, 0.0)
    mass.einstein_radius = 1.6
    mass.ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = 0.05
    shear.gamma_2 = 0.05

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass)
    field = af.Model(al.MassField, redshift=0.5, shear=shear)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

    return model


def build_analysis(dataset):
    import autolens as al

    return al.AnalysisImaging(dataset=dataset, use_jax=True)
