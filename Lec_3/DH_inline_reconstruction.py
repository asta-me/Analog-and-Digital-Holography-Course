#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: IN-LINE PARAMECIUM RECONSTRUCTION
#
# Author: Marco Astarita
# Date: November 8, 2026
#
# A real in-line hologram records interference between light scattered by the
# sample and unscattered illumination. Here the sample is Paramecium.
# Theory: Kostuk 7.4, 7.5.2 and 7.6; Fresnel TF as in Lectures 1 and 2.
#
# Data: Mike Hughes' lab, University of Kent, PyHoloscope (MIT license).
# https://github.com/MikeHughesKent/PyHoloscope/
# Source: notebooks/notebook01_inline_holography.ipynb and
# example_data/inline_paramecium/{paramecium.tif,background.tif}.
# Source parameters: wavelength 520 nm, pixel 4.8 um, depth 121 mm.
# The tutorial uses angular spectrum; this lesson retains Fresnel.
#
# Local inputs, prepared BEFORE the lesson:
#   inline_paramecium_raw.tif: unchanged camera hologram.
#   inline_paramecium_background.tif: measured without the sample.
#   inline_paramecium_clean.tif: RAW - BACKGROUND, float32, signed values.
# No offset, clipping, flat-field division or phase retrieval was applied.
# We load ONLY the clean contrast hologram; there is no subtraction step here.
# A contrast hologram is a real transmittance, not a nonnegative intensity.
#
# For a real input and R = 1, u(-z) = conjugate(u(+z)): identical intensities,
# opposite phases. The two foci are conjugate images, not two physical objects.
# Zero-order remnants and the twin image remain; phase is not quantitative.
# No phase unwrapping. Scan: 81 planes, nominal foci at slices 5/77, zero at 41.
# Only final intensity/phase ImageJ stacks are saved; plots stay in Matplotlib.
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
lam = 520e-9
dx = 4.8e-6
dy = dx
z_focus = 0.121
n_steps = 81
depths = (np.arange(n_steps) - n_steps//2) * (z_focus/36)
zoom_center = (420, 760)  # (row, col): above and right of the image centre
zoom_radius = 200

folder = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (folder / "inline_paramecium_clean.tif").exists():
    folder = Path.cwd() / "Lec_3"
hologram_path = folder / "inline_paramecium_clean.tif"
output_prefix = folder / "DH_inline_paramecium_final"

#%% 2. FRESNEL TRANSFER FUNCTION (SAME AS LECTURES 1 AND 2)
def propagate_Fresnel_TF(u1, dx, dy, lam, z):
    """Positive z propagates forward; negative z propagates backward."""
    N, M = u1.shape
    fx = fftshift(fftfreq(M, d=dx))
    fy = fftshift(fftfreq(N, d=dy))
    ux, uy = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lam * z * (ux**2 + uy**2))
    U1 = fftshift(fft2(ifftshift(u1)))
    return fftshift(ifft2(ifftshift(U1 * H)))

#%% 3. LOAD THE CLEAN CONTRAST HOLOGRAM
hologram = np.asarray(Image.open(hologram_path), dtype=np.float64)
hologram /= np.max(np.abs(hologram))
N, M = hologram.shape

plt.figure(figsize=(8, 6))
plt.imshow(hologram, cmap='gray')
plt.title("Paramecium: Background-Corrected In-Line Hologram")
plt.axis('off')
plt.colorbar(label="Signed contrast")
plt.show()

#%% 4. FOURIER SPECTRUM: NO SEPARATE SIDE PEAKS
spectrum = fftshift(fft2(ifftshift(hologram)))
gamma = 0.1
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spectrum)**gamma, cmap='gray')
plt.title(f"In-Line Hologram Spectrum |FFT|^{gamma}")
plt.axis('off')
plt.show()

#%% 5. RECONSTRUCT BOTH CONJUGATE IMAGES
R_recon = 1.0
field_after_hologram = R_recon * hologram
u_minus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, -z_focus)
u_plus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, +z_focus)
I_minus = np.abs(u_minus)**2
I_plus = np.abs(u_plus)**2
intensity_limit = np.percentile(I_plus, 99.5)
print(f"max |I(+z) - I(-z)| = {np.max(np.abs(I_plus-I_minus)):.1e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for col, (field, z) in enumerate(((u_minus, -z_focus), (u_plus, z_focus))):
    axes[0, col].imshow(np.abs(field)**2, cmap='gray', vmin=0, vmax=intensity_limit)
    axes[0, col].set_title(f"Intensity: z = {z*1e3:+.1f} mm")
    axes[1, col].imshow(np.angle(field), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, col].set_title(f"Wrapped Phase: z = {z*1e3:+.1f} mm")
    axes[0, col].axis('off')
    axes[1, col].axis('off')
plt.show()

#%% 6. ZOOM ON THE SAMPLE AND ITS TWIN-IMAGE HALO
row_center, col_center = zoom_center
crop = (slice(row_center-zoom_radius, row_center+zoom_radius),
        slice(col_center-zoom_radius, col_center+zoom_radius))
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
axes[0].imshow(hologram[crop], cmap='gray')
axes[0].set_title("Contrast Hologram")
for axis, intensity, z in ((axes[1], I_minus, -z_focus), (axes[2], I_plus, z_focus)):
    axis.imshow(intensity[crop], cmap='gray', vmin=0, vmax=intensity_limit)
    axis.set_title(f"Intensity: z = {z*1e3:+.1f} mm")
for axis in axes:
    axis.axis('off')
plt.show()

#%% 7. FINAL FULL-FIELD STACKS FOR IMAGEJ
labels = [f"z = {z*1e3:+.3f} mm" for z in depths]
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
    u_z = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, z)
    intensity_stack[index] = np.abs(u_z)**2
    phase_stack[index] = np.angle(u_z)
intensity_stack.flush()
phase_stack.flush()
del intensity_stack, phase_stack
print(f"Saved two {n_steps}-plane stacks: {output_prefix}")
