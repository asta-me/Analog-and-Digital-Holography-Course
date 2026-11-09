#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: OFF-AXIS TISSUE
#
# Author: Marco Astarita
# Date: November 8, 2026
#
# Real off-axis holography: tissue paper (cellulose fibres, NOT biological
# tissue) interferes with a tilted reference beam. The carrier separates the
# object, zero order and twin image in the Fourier plane. Select one side peak,
# shift it to the centre, apply a Gaussian mask and inverse FFT, then refocus
# with the Fresnel TF from Lectures 1 and 2.
# Theory: Kostuk 3.5.2, 7.3 and 7.4.
#
# Data: Mike Hughes' lab, University of Kent, PyHoloscope (MIT license).
# https://github.com/MikeHughesKent/PyHoloscope/
# Source: test/integration_tests/test data/tissue_paper_oa.tif and
# test/integration_tests/test_off_axis.py.
# Source parameters: wavelength 630 nm, effective sample pixel 0.3 um.
# The -30 to +10 um scan is a teaching range, not a calibrated sample thickness.
# The original example corrects background phase; here only Fourier filtering
# and Fresnel propagation are used, with no background input.
# The mask suppresses zero order and twin image, but does not calibrate slide
# tilt, aberrations or refractive index. Phase is wrapped, not optical thickness.
#
# Same procedure as DH_offaxis_paramecium.py; only sample settings differ.
# Only full-field intensity/phase ImageJ stacks are saved, no plot files.
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
lam = 630e-9
dx = 0.3e-6
dy = dx

sample_title = "Off-Axis Tissue"
hologram_path = "tissue_paper_oa.tif"
output_prefix = "DH_offaxis_tissue_output"

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
# hologram = |O + R|^2: the tilted reference creates a spatial carrier.
hologram /= np.max(np.abs(hologram))  # Normalised
N, M = hologram.shape                # Shape
center = (N//2, M//2)

plt.figure(figsize=(8, 6))
plt.imshow(hologram, cmap='gray')
plt.title(f"{sample_title}: Recorded Hologram")
plt.axis('off')
plt.colorbar(label="Normalized intensity")
plt.show()

fringe_radius = 60  # Zoom to resolve the reference-beam fringes [pixels]
fringe_crop = (slice(center[0]-fringe_radius, center[0]+fringe_radius),
               slice(center[1]-fringe_radius, center[1]+fringe_radius))
plt.figure(figsize=(8, 6))
plt.imshow(hologram[fringe_crop], cmap='gray')
plt.title(f"{sample_title}: Interference Fringes")
plt.axis('off')
plt.show()

#%% 4. FOURIER SPECTRUM: SELECT A SIDE PEAK
peak_position = (145, 872)  # (row, col) of the selected Fourier side peak
spectrum = fftshift(fft2(ifftshift(hologram)))  # Spectrum of the hologram
spectrum_module = np.abs(spectrum)
gamma = 0.1  # Gamma correction for spectrum visualization

plt.figure(figsize=(8, 6))
plt.imshow(spectrum_module**gamma, cmap='gray')
plt.plot(center[1], center[0], 'r.', markersize=10, label='Zero order')
plt.plot(peak_position[1], peak_position[0], 'r*', markersize=12, label='Selected side peak')
plt.title(f"{sample_title}: Hologram Spectrum |FFT|^{gamma}")
plt.legend(loc='lower right')
plt.axis('off')
plt.show()

#%% 5. SHIFT THE PEAK TO THE CENTRE AND FILTER
shift = np.array(center) - np.array(peak_position)  # Remove the carrier
spectrum_centered = np.roll(spectrum, shift, axis=(0, 1))

plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spectrum_centered)**gamma, cmap='gray')
plt.title(f"{sample_title}: Carrier Removed")
plt.axis('off')
plt.show()

sigma_mask = 50  # Gaussian width in the Fourier plane [pixels]
mask = gaussian_mask((N, M), center, sigma_mask)
plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='viridis')
plt.title(f"{sample_title}: Gaussian Mask, sigma = {sigma_mask} px")
plt.axis('off')
plt.show()

spectrum_filtered = mask * spectrum_centered  # Keep the selected object term
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spectrum_filtered)**gamma, cmap='gray')
plt.title(f"{sample_title}: Filtered Spectrum")
plt.axis('off')
plt.show()

#%% 6. RECOVER THE COMPLEX FIELD AND REFOCUS
u_object = fftshift(ifft2(ifftshift(spectrum_filtered)))  # Field at z = 0
I_object = np.abs(u_object)**2

plt.figure(figsize=(8, 6))
plt.imshow(I_object, cmap='gray')
plt.title(f"{sample_title}: Intensity, z = 0")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(np.angle(u_object), cmap='bwr', vmin=-np.pi, vmax=np.pi)
plt.title(f"{sample_title}: Wrapped Phase, z = 0")
plt.axis('off')
plt.colorbar(label="Phase [rad]")
plt.show()

z_focus = -10e-6  # Refocus distance for the selected sideband [m]
u_focus = propagate_Fresnel_TF(u_object, dx, dy, lam, z_focus)
I_focus = np.abs(u_focus)**2

plt.figure(figsize=(8, 6))
plt.imshow(I_focus, cmap='gray')
plt.title(f"{sample_title}: Intensity, z = {z_focus*1e6:+.1f} um")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(np.angle(u_focus), cmap='bwr', vmin=-np.pi, vmax=np.pi)
plt.title(f"{sample_title}: Wrapped Phase, z = {z_focus*1e6:+.1f} um")
plt.axis('off')
plt.colorbar(label="Phase [rad]")
plt.show()


#%% 7. OUTPUT FULL-FIELD STACKS FOR IMAGEJ
# Scan a small defocusing interval around the interesting focus.
depth_start = z_focus - 20e-6  # [m]
depth_end = z_focus + 20e-6    # [m]
n_steps = 41
depths = np.linspace(depth_start, depth_end, n_steps)

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
