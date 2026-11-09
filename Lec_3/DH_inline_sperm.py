#%%
# =============================================================================
# DIGITAL HOLOGRAPHY: INLINE SPERM
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
# Authors' Gabor baseline: sqrt(I) estimates the camera-plane amplitude with
# zero initial phase; this alone does not remove the twin image.
# Their full YOSO method uses a neural network to generate an additional
# defocused hologram, then Gerchberg-Saxton (GS) phase retrieval to suppress
# artifacts. Compare their results in Figure 6 of the paper during the lesson.
# Here we use the lecture's amplitude-hologram model: transmittance t = I,
# illuminated by a uniform reconstruction beam R_recon = 1, then Fresnel
# propagation. We expect visible zero-order and twin-image artifacts.
# No neural network or GS is used here; sqrt(I) remains a commented alternative.
# Propagation uses the original camera grid without padding; edge artifacts
# can remain. The lesson focuses on the interior regions of the image.
# The short scan uses negative z around -70 um; the optional wide scan shows
# both conjugate images. Positive d = -z is the back-propagation magnitude.
# A real zero-phase input gives identical intensities at +/-z and opposite
# phases; this does not imply two physical copies of the sample.
# Different apparent best-focus planes are qualitative, not calibrated depths.
# Zoom positions A/B/C approximately follow Fig. 6, not labelled ground truth.
# Only full-field intensity/phase ImageJ stacks are saved; no plot files.
#
# Requires NumPy, Matplotlib and tifffile. Run #%% cells in order or run this
# file as a regular script, with Lec_3 as the working directory.
# =============================================================================

#%% 0. IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS
lam = 635e-9       # Wavelength [m]
camera_pixel = 5.5e-6
magnification = 20
dx = camera_pixel / magnification  # Effective pixel size at the sample [m]
dy = dx

hologram_path = "sperm_inline_holo.tif"

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
# hologram = |O + R|^2: recorded intensity; no background image is used here.
hologram /= np.max(np.abs(hologram))  # Normalised
N, M = hologram.shape                 # Shape

plt.figure(figsize=(8, 6))
plt.imshow(hologram, cmap='gray')
plt.title("Inline Sperm: Raw Hologram")
plt.axis('off')
plt.colorbar(label="Normalized intensity")
plt.show()

#%% 4. INSPECT THE FOURIER SPECTRUM
spectrum = fftshift(fft2(ifftshift(hologram)))  # Spectrum of the hologram
spectrum_module = np.abs(spectrum)
gamma = 0.1  # Gamma correction for spectrum visualization

plt.figure(figsize=(8, 6))
plt.imshow(spectrum_module**gamma, cmap='gray')
plt.title(f"Inline Sperm: Hologram Spectrum |FFT|^{gamma}")
plt.axis('off')
plt.show()

#%% 5. ILLUMINATE THE HOLOGRAM WITH THE RECONSTRUCTION BEAM
# The recorded hologram acts as an amplitude transmittance, as in Lecture 2.
R_recon = 1.0
field_after_hologram = R_recon * hologram
# Authors' zero-phase Gabor baseline (alternative, not the full YOSO method):
# field_after_hologram = np.sqrt(hologram)

#%% 6. RECONSTRUCT BOTH CONJUGATE IMAGES
# z_focus = 70e-6   # Nominal defocus magnitude [m]; back-propagation uses -z_focus
z_focus = 68.6e-6   # Focal position of a cell
u_minus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, -z_focus)
u_plus = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, +z_focus)
I_minus = np.abs(u_minus)**2
I_plus = np.abs(u_plus)**2
print(f"max |I(+z) - I(-z)| = {np.max(np.abs(I_plus-I_minus)):.1e}")

plt.figure(figsize=(8, 6))
plt.imshow(I_minus, cmap='gray')
plt.title(f"Inline Sperm: Intensity, z = {-z_focus*1e6:+.1f} um")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(I_plus, cmap='gray')
plt.title(f"Inline Sperm: Intensity, z = {z_focus*1e6:+.1f} um")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(np.angle(u_minus), cmap='bwr', vmin=-np.pi, vmax=np.pi)
plt.title(f"Inline Sperm: Wrapped Phase, z = {-z_focus*1e6:+.1f} um")
plt.axis('off')
plt.colorbar(label="Phase [rad]")
plt.show()

#%% 7. ZOOM ON THE SAMPLE AND ITS TWIN-IMAGE HALO
# Approximate Figure 6 positions (row, col): A=(920,910), B=(495,425), C=(160,960).
zoom_center = (160, 960)  # Region C; change this to inspect A or B
zoom_radius = 55

row_center, col_center = zoom_center
crop = (slice(row_center-zoom_radius, row_center+zoom_radius),
    slice(col_center-zoom_radius, col_center+zoom_radius))

plt.figure(figsize=(8, 6))
plt.imshow(hologram[crop], cmap='gray')
plt.title("Inline Sperm: Recorded Hologram (Zoom)")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(I_minus[crop], cmap='gray')
plt.title(f"Inline Sperm: Intensity, z = {-z_focus*1e6:+.1f} um")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(I_plus[crop], cmap='gray')
plt.title(f"Inline Sperm: Intensity, z = {z_focus*1e6:+.1f} um")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(np.angle(u_minus[crop]), cmap='bwr', vmin=-np.pi, vmax=np.pi)
plt.title(f"Inline Sperm: Wrapped Phase, z = {-z_focus*1e6:+.1f} um")
plt.axis('off')
plt.colorbar(label="Phase [rad]")
plt.show()

#%% 8. FULL-FIELD RECONSTRUCTION STACKS FOR IMAGEJ
# To see both conjugate images, use this wide range instead of the short one.
depth_start = -100e-6  # [m]
depth_end = +100e-6    # [m]
n_steps = 81
depths = np.linspace(depth_start, depth_end, n_steps)

# Alternatively, scan a little defocusing around one interesting focus.
depth_start = -z_focus - 8e-6  # [m]
depth_end = -z_focus + 8e-6    # [m]
n_steps = 51
depths = np.linspace(depth_start, depth_end, n_steps)

output_prefix = "DH_inline_sperm_output"
labels = [f"z = {z*1e6:+.1f} um (d = {-z*1e6:.1f} um)" for z in depths]
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
    u = propagate_Fresnel_TF(field_after_hologram, dx, dy, lam, z)
    intensity_stack[index] = np.abs(u)**2
    phase_stack[index] = np.angle(u)
intensity_stack.flush()
phase_stack.flush()
del intensity_stack, phase_stack
print(f"Saved two {n_steps}-plane stacks: {output_prefix}")
# %%
