#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: OFF-AXIS PARAMECIUM RECONSTRUCTION
#
# Author: Marco Astarita
# Date: November 8, 2026
#
# Real off-axis holography: light transmitted by Paramecium interferes with a
# tilted reference beam. The carrier separates object, zero order and twin
# image in the Fourier plane. Select one side peak and shift it to the centre
# before applying a Gaussian mask and an inverse FFT. Then refocus the field
# with the Fresnel TF from Lectures 1 and 2.
# Theory: Kostuk 3.5.2, 7.3 and 7.4.
#
# Data: Mike Hughes' lab, University of Kent, PyHoloscope (MIT license).
# https://github.com/MikeHughesKent/PyHoloscope/
# Source: test/integration_tests/test data/paramecium_oa_oof.tif and
# examples/off_axis_example_refocusing.py.
# Source wavelength 630 nm, effective pixel 1 um, refocus magnitude 1129 um.
# The chosen Fourier sideband fixes the field conjugation: here focus uses
# z = -1129 um; selecting the conjugate sideband reverses the propagation sign.
# The original example uses angular spectrum and background phase correction;
# this lesson uses only Fresnel and Fourier filtering, with no background input.
# The mask suppresses zero order and twin image, but does not calibrate slide
# tilt, aberrations or refractive index. Phase is wrapped, not optical thickness.
# No organelles are independently identified in this dataset.
#
# Same procedure as DH_offaxis_reconstruction.py; only settings differ.
# Only final full-field intensity/phase ImageJ stacks are saved, no plot files.
# Requires NumPy, Matplotlib, Pillow, tifffile. Run #%% cells in order or run
# this file as a regular script, from Lec_3 or the repository root.
# =============================================================================

#%% 0. IMPORTS
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS AND INPUT
sample_title = "Paramecium"
hologram_name = "paramecium_oa_oof.tif"
output_name = "DH_offaxis_paramecium_final"
lam = 630e-9
dx = 1e-6
dy = dx
peak_position = (251, 1064)  # (row, col) of the selected Fourier side peak
sigma_mask = 75
z_focus = -1129e-6
depth_half_range = 200e-6
n_steps = 41
depths = np.linspace(z_focus-depth_half_range, z_focus+depth_half_range, n_steps)
zoom_center = (350, 650)
zoom_radius = 200

folder = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (folder / hologram_name).exists():
    folder = Path.cwd() / "Lec_3"
hologram_path = folder / hologram_name
output_prefix = folder / output_name

#%% 2. FRESNEL TRANSFER FUNCTION AND GAUSSIAN MASK
def propagate_Fresnel_TF(u1, dx, dy, lam, z):
    """Positive z propagates forward; negative z propagates backward."""
    N, M = u1.shape
    fx = fftshift(fftfreq(M, d=dx))
    fy = fftshift(fftfreq(N, d=dy))
    ux, uy = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lam * z * (ux**2 + uy**2))
    U1 = fftshift(fft2(ifftshift(u1)))
    return fftshift(ifft2(ifftshift(U1 * H)))


def gaussian_mask(shape, center, sigma):
    """Soft Fourier window centred at (row, col), avoiding a sharp cutoff."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    radius_squared = (rows-center[0])**2 + (cols-center[1])**2
    return np.exp(-radius_squared/(2*sigma**2))

#%% 3. LOAD THE RECORDED HOLOGRAM AND INSPECT THE FRINGES
hologram = np.asarray(Image.open(hologram_path), dtype=np.float64)
hologram /= hologram.max()
N, M = hologram.shape
center = (N//2, M//2)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
axes[0].imshow(hologram, cmap='gray')
axes[0].set_title(f"Off-Axis Hologram: {sample_title}")
axes[1].imshow(hologram[center[0]-60:center[0]+60, center[1]-60:center[1]+60], cmap='gray')
axes[1].set_title("Interference Fringes (Carrier)")
for axis in axes:
    axis.axis('off')
plt.show()

#%% 4. FOURIER SPECTRUM: SELECT A SIDE PEAK
spectrum = fftshift(fft2(ifftshift(hologram)))
gamma = 0.1
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spectrum)**gamma, cmap='gray')
plt.plot(center[1], center[0], 'r.', markersize=10, label='Zero order')
plt.plot(peak_position[1], peak_position[0], 'r*', markersize=12, label='Selected side peak')
plt.title(f"Off-Axis Spectrum |FFT|^{gamma}: {sample_title}")
plt.legend(loc='lower right')
plt.axis('off')
plt.show()

#%% 5. SHIFT THE PEAK TO THE CENTRE AND FILTER
shift = np.array(center) - np.array(peak_position)
spectrum_centered = np.roll(spectrum, shift, axis=(0, 1))
mask = gaussian_mask((N, M), center, sigma_mask)
spectrum_filtered = mask * spectrum_centered

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
axes[0].imshow(np.abs(spectrum_centered)**gamma, cmap='gray')
axes[0].set_title("Carrier Removed")
axes[1].imshow(mask, cmap='viridis')
axes[1].set_title(f"Gaussian Mask: sigma = {sigma_mask} px")
axes[2].imshow(np.abs(spectrum_filtered)**gamma, cmap='gray')
axes[2].set_title("Selected Object Spectrum")
for axis in axes:
    axis.axis('off')
plt.show()

#%% 6. RECOVER THE COMPLEX FIELD AND REFOCUS
u_object = fftshift(ifft2(ifftshift(spectrum_filtered)))
u_focus = propagate_Fresnel_TF(u_object, dx, dy, lam, z_focus)
intensity_limit = np.percentile(np.abs(u_focus)**2, 99.5)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for col, (field, z) in enumerate(((u_object, 0.0), (u_focus, z_focus))):
    axes[0, col].imshow(np.abs(field)**2, cmap='gray', vmin=0, vmax=intensity_limit)
    axes[0, col].set_title(f"Intensity: z = {z*1e6:+.1f} um")
    axes[1, col].imshow(np.angle(field), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, col].set_title(f"Wrapped Phase: z = {z*1e6:+.1f} um")
    axes[0, col].axis('off')
    axes[1, col].axis('off')
plt.show()

#%% 7. ZOOM ON THE REFOCUSED SAMPLE
row_center, col_center = zoom_center
crop = (slice(row_center-zoom_radius, row_center+zoom_radius),
        slice(col_center-zoom_radius, col_center+zoom_radius))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].imshow(np.abs(u_focus[crop])**2, cmap='gray', vmin=0, vmax=intensity_limit)
axes[0].set_title(f"Refocused Intensity: {sample_title}")
axes[1].imshow(np.angle(u_focus[crop]), cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[1].set_title("Wrapped Phase")
for axis in axes:
    axis.axis('off')
plt.show()

#%% 8. FINAL FULL-FIELD STACKS FOR IMAGEJ
labels = [f"z = {z*1e6:+.2f} um" for z in depths]
metadata = {"axes": "ZYX", "Labels": labels, "unit": "um",
            "spacing": (depths[1]-depths[0])*1e6}
resolution = (1/(dx*1e6), 1/(dy*1e6))
intensity_stack = tifffile.memmap(f"{output_prefix}_intensity_stack.tif",
                                 shape=(n_steps, N, M), dtype=np.float32,
                                 imagej=True, metadata=metadata, resolution=resolution)
phase_stack = tifffile.memmap(f"{output_prefix}_phase_stack.tif",
                             shape=(n_steps, N, M), dtype=np.float32,
                             imagej=True, metadata=metadata, resolution=resolution)
for index, z in enumerate(depths):
    u_z = propagate_Fresnel_TF(u_object, dx, dy, lam, z)
    intensity_stack[index] = np.abs(u_z)**2
    phase_stack[index] = np.angle(u_z)
intensity_stack.flush()
phase_stack.flush()
del intensity_stack, phase_stack
print(f"Saved two {n_steps}-plane stacks: {output_prefix}")