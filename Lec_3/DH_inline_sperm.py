#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: IN-LINE RECONSTRUCTION OF SPERM CELLS
#
# Author: Marco Astarita
# Date: November 8, 2026
#
# An experimental in-line hologram contains sperm cells. We reconstruct a
# complex field using one recorded intensity image, and scan the propagation
# distance to explore which structures come into focus.
# The zero order and the twin image remain superimposed: the reconstructed
# phase is not a quantitative phase map and this is not 3D tomography.
# Theory: Kostuk 7.4, 7.5.2 and 7.6; Fresnel TF as in Lectures 1 and 2.
#
# Dataset: Winnik et al., YOSO dataset (CC BY 4.0), first hologram i_ccd1.
# https://doi.org/10.5281/zenodo.19690495
# https://doi.org/10.1088/2515-7647/ae99bb
# Source file: data.zip/data/sperm.h5, dataset i_ccd1 (1408 x 1400, float64).
# sperm_inline_holo.tif is an unchanged export of that measured intensity.
# No background image is used. No artifact-free reference map is supplied here.
# The sample species is not specified in the dataset description.
# Lens-based acquisition: 635 nm, camera pixel 5.5 um, 20x objective, NA 0.46.
# Effective sample pixel: 5.5/20 = 0.275 um; nominal defocus: 70 um.
# The sqrt(I), zero-phase input follows the authors' Gabor baseline; it differs
# from the contrast-transmittance model in the Paramecium lesson.
# The published YOSO reconstructions suppress artifacts using AI-assisted GS;
# compare with Figure 6 of the paper. Here there is NO GS or neural network.
# We scan only negative z: positive d = -z is the back-propagation magnitude.
# Different apparent best-focus planes are qualitative, not calibrated depths.
# Regions A/B/C approximately follow Fig. 6, not independently labelled cells.
# Only final full-field intensity/phase ImageJ stacks are saved; no plot files.
#
# Requires NumPy, Matplotlib and tifffile. Run #%% cells in order or run this
# file as a regular script, from Lec_3 or the repository root.
# =============================================================================

#%% 0. IMPORTS
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS
lam = 635e-9       # Wavelength [m]
camera_pixel = 5.5e-6
magnification = 20
dx = camera_pixel / magnification  # Effective pixel size at the sample [m]
dy = dx
d_focus = 70e-6    # Nominal back-propagation distance [m]
padding = 128      # Edge replication on each side [pixels]

folder = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (folder / "sperm_inline_holo.tif").exists():
    folder = Path.cwd() / "Lec_3"
hologram_path = folder / "sperm_inline_holo.tif"
output_dir = folder

#%% 2. PROPAGATION ALGORITHM: FRESNEL TRANSFER FUNCTION

def propagate_Fresnel_TF(u1, dx, dy, lam, z):
    """Propagate a 2D field by z; negative z propagates backwards."""
    N, M = u1.shape
    fx = fftshift(fftfreq(M, d=dx))
    fy = fftshift(fftfreq(N, d=dy))
    ux, uy = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lam * z * (ux**2 + uy**2)).astype(np.complex64)
    U1 = fftshift(fft2(ifftshift(u1)))
    u2 = fftshift(ifft2(ifftshift(U1 * H)))
    return u2

#%% 3. LOAD AND DISPLAY THE CAMERA HOLOGRAM
hologram = tifffile.imread(hologram_path).astype(np.float32)
N, M = hologram.shape
hologram /= np.median(hologram)
print(f"Hologram: {N} x {M} pixels; effective pixel size = {dx*1e6:.3f} um")
print(f"Wavelength = {lam*1e9:.0f} nm; nominal focus distance = {d_focus*1e6:.0f} um")

regions = {"A": (920, 910), "B": (495, 425), "C": (160, 960)}
radius = 55
plt.figure(figsize=(9, 6))
plt.imshow(hologram, cmap='gray', vmin=np.percentile(hologram, 1), vmax=np.percentile(hologram, 99))
for name, (row_center, col_center) in regions.items():
    plt.gca().add_patch(Rectangle((col_center-radius, row_center-radius), 2*radius, 2*radius,
                                  fill=False, edgecolor='red', linewidth=1.5))
    plt.text(col_center-radius, row_center-radius-10, name, color='red', fontsize=12)
plt.title("Recorded In-Line Hologram: Sperm Sample")
plt.axis('off')
plt.colorbar(label="Intensity / median")
plt.show()

#%% 4. INSPECT THE FOURIER SPECTRUM
spectrum = fftshift(fft2(ifftshift(hologram)))
gamma = 0.1
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spectrum)**gamma, cmap='gray')
plt.title(f"In-Line Hologram Spectrum |FFT|^{gamma}")
plt.axis('off')
plt.show()

#%% 5. FIELD ASSUMPTION AT THE HOLOGRAM PLANE
# The camera measures intensity, not the complex field. As in the authors'
# Gabor baseline, use sqrt(I) as amplitude and assume zero initial phase.
# This differs from the I*R hologram-transmittance model of the other example.
field_after_hologram = np.sqrt(hologram)
field_padded = np.pad(field_after_hologram, padding, mode='edge')

#%% 6. RECONSTRUCTION AT THE NOMINAL DISTANCE
# d is the positive back-propagation distance; z = -d in our convention.
u_padded = propagate_Fresnel_TF(field_padded, dx, dy, lam, -d_focus)
u_focus = u_padded[padding:-padding, padding:-padding]
I_focus = np.abs(u_focus)**2
phase_focus = np.angle(u_focus)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(I_focus, cmap='gray', vmin=np.percentile(I_focus, 1), vmax=np.percentile(I_focus, 99))
plt.title(f"Intensity at d = {d_focus*1e6:.0f} um")
plt.axis('off')
plt.colorbar(label="Relative intensity")
plt.subplot(1, 2, 2)
plt.imshow(phase_focus, cmap='twilight', vmin=-0.4, vmax=0.4)
plt.title(f"Wrapped Gabor Phase at d = {d_focus*1e6:.0f} um")
plt.axis('off')
plt.colorbar(label="Phase [rad]")
plt.tight_layout()
plt.show()

#%% 7. THREE REGIONS AT DIFFERENT PROPAGATION DISTANCES
# Approximate locations of regions A, B and C in Fig. 6 of Winnik et al.
# Other particles are also present; these positions are not dataset labels.
distances_um = [62, 70, 78]

plt.figure(figsize=(10, 9))
for col, distance_um in enumerate(distances_um):
    u_padded = propagate_Fresnel_TF(field_padded, dx, dy, lam, -distance_um*1e-6)
    u = u_padded[padding:-padding, padding:-padding]
    for row, (name, (row_center, col_center)) in enumerate(regions.items()):
        crop = (slice(row_center-radius, row_center+radius),
                slice(col_center-radius, col_center+radius))
        plt.subplot(len(regions), len(distances_um), row*len(distances_um)+col+1)
        plt.imshow(np.angle(u[crop]), cmap='viridis', vmin=-0.4, vmax=0.4)
        plt.title(f"{name}: d = {distance_um} um")
        plt.axis('off')
plt.suptitle("One Hologram: Refocusing at Different Depths")
plt.tight_layout()
plt.show()

#%% 8. FULL-FIELD RECONSTRUCTION STACKS FOR IMAGEJ
depth_start = -100e-6
depth_end = -40e-6
n_steps = 61
depths = np.linspace(depth_start, depth_end, n_steps)
labels = [f"z = {z*1e6:+.1f} um (d = {-z*1e6:.1f} um)" for z in depths]
metadata = {"axes": "ZYX", "Labels": labels, "unit": "um",
            "spacing": (depths[1]-depths[0])*1e6}
resolution = (1/(dx*1e6), 1/(dy*1e6))
output_prefix = output_dir / "DH_inline_sperm_final"
intensity_stack = tifffile.memmap(f"{output_prefix}_intensity_stack.tif",
                                 shape=(n_steps, N, M), dtype=np.float32,
                                 imagej=True, metadata=metadata, resolution=resolution)
phase_stack = tifffile.memmap(f"{output_prefix}_phase_stack.tif",
                             shape=(n_steps, N, M), dtype=np.float32,
                             imagej=True, metadata=metadata, resolution=resolution)
for index, z in enumerate(depths):
    u_padded = propagate_Fresnel_TF(field_padded, dx, dy, lam, z)
    u = u_padded[padding:-padding, padding:-padding]
    intensity_stack[index] = np.abs(u)**2
    phase_stack[index] = np.angle(u)
intensity_stack.flush()
phase_stack.flush()
del intensity_stack, phase_stack
print(f"Saved two {n_steps}-plane stacks: {output_prefix}")