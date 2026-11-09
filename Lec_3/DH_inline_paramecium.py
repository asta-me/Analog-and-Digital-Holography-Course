#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: INLINE PARAMECIUM
#
# Author: Marco Astarita
# Date: November 8, 2026
#
# A real in-line hologram records interference between light scattered by the
# sample and unscattered illumination. Here the sample is a paramecium.
# Theory: Kostuk 7.4, 7.5.2 and 7.6; Fresnel TF as in Lectures 1 and 2.
#
# Data: Mike Hughes' lab, University of Kent, PyHoloscope (MIT license).
# https://github.com/MikeHughesKent/PyHoloscope/
# Source: notebooks/notebook01_inline_holography.ipynb and
# example_data/inline_paramecium/{paramecium.tif,background.tif}.
# Source parameters: wavelength 520 nm, pixel 4.8 um, depth 121 mm.
# The tutorial uses angular spectrum; this lesson retains Fresnel.
#
# Local inputs:
#   inline_paramecium_raw.tif: unchanged camera hologram.
#   inline_paramecium_background.tif: measured without the sample.
# Subtract the measured background at the start of the lesson, then propagate
# the resulting contrast hologram. Illumination and exposure must match.
# No flat-field division or phase retrieval is performed.
#
# For a real input and R = 1, u(-z) = conjugate(u(+z)): identical intensities,
# opposite phases. The two foci are conjugate images, not two physical objects.
# Zero-order remnants and the twin image remain; phase is not quantitative.
# No phase unwrapping. Scan from negative to positive z across both foci.
# Only intensity/phase ImageJ stacks are saved; plots stay in Matplotlib.
# Requires NumPy, Matplotlib, Pillow, tifffile. Run #%% cells in order or run
# this file as a regular script, with Lec_3 as the working directory.
# =============================================================================

#%% 0. IMPORTS
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

hologram_path = "inline_paramecium_raw.tif"
background_path = "inline_paramecium_background.tif"
output_prefix = "DH_inline_paramecium_output"

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

#%% 3. LOAD THE RAW HOLOGRAM AND SUBTRACT THE BACKGROUND
hologram_raw = np.asarray(Image.open(hologram_path), dtype=np.float64)
# hologram_raw = |O + R|^2 = |O|^2 + |R|^2 + O*R + OR*
background = np.asarray(Image.open(background_path), dtype=np.float64)
# background ~= |R|^2: same acquisition with the sample removed.
hologram = hologram_raw - background
# raw - background ~= |O|^2 + O*conj(R) + conj(O)*R

hologram /= np.max(np.abs(hologram)) # Normalised
N, M = hologram.shape                # Shape

plt.figure(figsize=(8, 6))
plt.imshow(hologram_raw, cmap='gray')
plt.title("Inline Paramecium: Raw Hologram")
plt.axis('off')
plt.colorbar(label="Signed contrast")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(background, cmap='gray')
plt.title("Inline Paramecium: Background")
plt.axis('off')
plt.colorbar(label="Signed contrast")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(hologram, cmap='gray')
plt.title("Inline Paramecium: Background-Corrected Hologram")
plt.axis('off')
plt.colorbar(label="Signed contrast")
plt.show()

#%% 4. FOURIER SPECTRUM: NO SEPARATE SIDE PEAKS
spectrum = fftshift(fft2(ifftshift(hologram)))      #Spectrum of the hologram
spectrum_module = np.abs(spectrum)
gamma = 0.1     # Gamma correction for spectrum visualization

plt.figure(figsize=(8, 6))
plt.imshow(spectrum_module**gamma, cmap='gray')
plt.title(f"Inline Paramecium: Hologram Spectrum |FFT|^{gamma}")
plt.axis('off')
plt.show()

#%% 5. RECONSTRUCT BOTH CONJUGATE IMAGES
R_recon = 1.0
field_after_hologram = R_recon * hologram
u_minus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, -z_focus)
u_plus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, +z_focus)
I_minus = np.abs(u_minus)**2
I_plus = np.abs(u_plus)**2
# intensity_limit = np.percentile(I_plus, 99.5)
print(f"max |I(+z) - I(-z)| = {np.max(np.abs(I_plus-I_minus)):.1e}")


plt.figure(figsize=(8, 6))
plt.imshow(I_minus, cmap='gray')
plt.title(f"Inline Paramecium: Intensity virtual, z = {-z_focus*1e3:+.1f} mm")
plt.axis('off')

plt.figure(figsize=(8, 6))
plt.imshow(I_plus, cmap='gray')
plt.title(f"Inline Paramecium: Intensity real, z = {z_focus*1e3:+.1f} mm")
plt.axis('off')
plt.show()

#%% 6. ZOOM ON THE SAMPLE AND ITS TWIN-IMAGE HALO
zoom_center = (300,700)  # (row, col): above and right of the image centre
zoom_radius = 250

row_center, col_center = zoom_center
crop = (slice(row_center-zoom_radius, row_center+zoom_radius),
        slice(col_center-zoom_radius, col_center+zoom_radius))


plt.figure(figsize=(8, 6))
plt.imshow(hologram[crop], cmap='gray')
plt.title("Inline Paramecium: Recorded Hologram (Zoom)")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(I_minus[crop], cmap='gray')
plt.title(f"Inline Paramecium: Intensity virtual, z = {-z_focus*1e3:+.1f} mm")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(I_plus[crop], cmap='gray')
plt.title(f"Inline Paramecium: Intensity real, z = {z_focus*1e3:+.1f} mm")
plt.axis('off')
plt.show()

#%% 7. OUTPUT FULL-FIELD STACKS FOR IMAGEJ
# To see real and virtual images, we create full-field stacks over the depth range
depth_start = -0.135  # [m]
depth_end = +0.135    # [m]
n_steps = 81
depths = np.linspace(depth_start, depth_end, n_steps)

# Alternatively we can see the defocusing in deep
depth_start = z_focus - 0.025  # [m]
depth_end = z_focus + 0.025    # [m]
n_steps = 30
depths = np.linspace(depth_start, depth_end, n_steps)

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

# %%
