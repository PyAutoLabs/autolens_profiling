"""Extract the real source-plane DelaunayNN mesh grid from the HST breakdown cell.

Reuses the minimum of scripts/imaging/likelihood_breakdown/delaunay_nn.py needed
to reach ``relocated_mesh_grid`` (1500 Hilbert vertices traced to the source
plane and border-relocated). Writes it to --out as .npy.

Run from the autolens_profiling root.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts" / "misc"))

import autofit as af
import autolens as al
import jax.numpy as jnp
from simulators.imaging import INSTRUMENTS

from _adapt_image_util import adapt_image_for_dataset

instrument = "hst"
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / instrument

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask_radius = 3.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=4, over_sample_size_pixelization=1)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size, over_sample_size_pixelization=1
)

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

n_mesh_vertices = 1500
image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)
print("mesh vertices placed:", image_plane_mesh_grid.shape[0])

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=60, centre_prior_is_uniform=True
)
mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_ell[0], sigma=0.01)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_ell[1], sigma=0.01)
shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

mesh = al.mesh.DelaunayNN(pixels=n_mesh_vertices, areas_factor=0.5, zeroed_pixels=0)
regularization = al.reg.ConstantSplit(coefficient=1.0)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
tracer = al.Tracer(galaxies=list(instance.galaxies))

from autoarray.inversion.mesh.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)
traced_source_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)[-1]
traced_mesh_source = tracer.traced_grid_2d_list_from(
    grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
)[-1]
relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
    grid=traced_source_grid, mesh_grid=traced_mesh_source
)

pts = np.asarray(relocated_mesh_grid.array, dtype=np.float64)
print("relocated mesh grid:", pts.shape, pts.dtype)
np.save(args.out, pts)
print(
    "areas_factor:",
    mesh.areas_factor,
    "max_neighbors:",
    mesh.max_neighbors,
    "max_cavity_triangles:",
    mesh.max_cavity_triangles,
)
print("wrote", args.out)
