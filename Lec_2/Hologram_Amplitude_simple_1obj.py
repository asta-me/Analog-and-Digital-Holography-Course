#%%
# =============================================================================
# DIGITAL HOLOGRAPHY SIMULATION: SINGLE-PLANE OBJECT
#
# Author: Marco Astarita
# Date: 2025-10-16
#
# Purpose:
# This script simulates the recording and reconstruction of a digital hologram
# from a 2D object field. It employs the Transfer Function (TF) method
# for Fresnel propagation. The setup can be configured for on-axis or
# off-axis holography.

# Some artifacts are present and should be improved.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image

#%%
# --- PROPAGATOR DEFINITION ---
def propagate_TF(u_in, dx, lam, z):
    N, M = u_in.shape
    # Frequency coordinates
    fx = fftshift(fftfreq(M, d=dx))
    fy = fftshift(fftfreq(N, d=dx))
    fX, fY = np.meshgrid(fx, fy)
    # Fresnel Transfer Function (TF)
    H = np.exp(-1j * np.pi * lam * z * (fX**2 + fY**2))
    # Convolution via frequency domain multiplication
    U_in = fftshift(fft2(ifftshift(u_in)))
    U_out = U_in * H
    u_out = fftshift(ifft2(ifftshift(U_out)))
    return u_out

#%%
# --- 1. SIMULATION PARAMETERS ---
# Grid Parameters
N = 2048            # Number of samples
M = 2048
dx = 5 * 1e-6       # Sampling interval [m]
dy = dx

# Physical Parameters
lam = 633 * 1e-9                # Wavelength [m]
k = 2 * np.pi / lam             # Wavenumber [rad/m]
dist = 0.2                    # Object distance [m]
# dist = 0.005                    # ??? Let's see

# Holography Parameters
theta_x_rad = 0                   # On-axis configuration (no tilt)
theta_x_rad = np.deg2rad(0.55)    # Off-axis angles 
theta_y_rad = theta_x_rad

pupil_radius_factor = 0.5       # Reference beam pupil size relative to the grid radius.

#%%
# --- 2. OBJECT DEFINITION ---
# Load object amplitude from a file.
# 'm_big_centered.bmp' is a 2048x2048 grayscale image.
obj_amp = np.array(Image.open("m_big_centered.bmp")) / 255.0

# Apply a random phase to simulate a real diffusing object
rand_phase = np.random.rand(M, N) * 2 * np.pi
rand_phase *=0

u1 = obj_amp * np.exp(1j * rand_phase)

plt.figure()
plt.imshow(obj_amp, cmap='gray')
plt.title('Object Amplitude')
plt.axis('off')

plt.figure()
plt.imshow(np.angle(u1)*obj_amp, cmap='hsv')
plt.colorbar()
plt.title('Object Phase')
plt.axis('off')

#%%
# --- 3. HOLOGRAM RECORDING ---

# Propagate the object wave to the hologram plane
O = propagate_TF(u1, dx, lam, dist)

# Define the reference wave (R)
x_vec = np.linspace(-1, 1, M) * dx * M/2
y_vec = np.linspace(-1, 1, N) * dy * N/2
X, Y = np.meshgrid(x_vec, y_vec)

# Create a tilted plane wave limited by a circular pupil
pupil_mask = (np.sqrt(X**2 + Y**2) <= pupil_radius_factor * (M * dx / 2)).astype(float)
#Apply off axis tilt
R = pupil_mask * np.exp(1j * k * (X * np.sin(theta_x_rad) + Y * np.sin(theta_y_rad)))

# The hologram is the recorded intensity of the interference pattern
balance_factor = 5               # Factor to balance the object and reference wave intensities
O *= balance_factor              # Scale the object wave to balance with the reference wave intensity
I = np.abs(O + R)**2             # Recorded intensity of the interference pattern

hologram = I                     # This is a transmittance hologram
# hologram -= np.mean(hologram) # Optional: remove DC background

plt.figure()
plt.imshow(hologram, cmap='gray')
plt.title('Recorded Hologram')
#%%
# --- 4. HOLOGRAPHIC RECONSTRUCTION ---

# The hologram is illuminated by the reconstruction beam (R_recon)
R_recon = R
field_after_hologram = R_recon * hologram

# Propagate to reconstruct the real and virtual images
u_real = propagate_TF(field_after_hologram, dx, lam, dist)
u_virtual = propagate_TF(field_after_hologram, dx, lam, -dist)

# Calculate intensities for display
I_real = np.abs(u_real)**2
I_virtual = np.abs(u_virtual)**2
# Normalised intensities
I_real /= np.max(I_real)
I_virtual /= np.max(I_virtual)

#%%
# --- 5. VISUALIZATION ---

# Visualization note: for linear scale plots, clipping with 'vmax' is
# useful to prevent the bright zero-order from saturating the colormap.
# A value like vmax=0.75 often works well.
I_threshold = 0.1

plt.figure()
plt.imshow(I_real, cmap='hot', vmax=I_threshold)
plt.title('Reconstructed Real Image')
plt.axis('off')
plt.colorbar(label="Normalized Intensity")

plt.figure()
plt.imshow(I_virtual, cmap='hot', vmax=I_threshold)
plt.title('Reconstructed Virtual Image')
plt.axis('off')
plt.colorbar(label="Normalized Intensity")
plt.show()

# %%
# --- 6. OPTIONAL: RECONSTRUCTION STACK FOR IMAGEJ ---
# import tifffile

# depths = np.linspace(-1.25 * dist, 1.25 * dist, 41)
# intensity_stack = []
# labels = []

# for z in depths:
#     field_z = propagate_TF(field_after_hologram, dx, lam, z)
#     intensity_stack.append(np.abs(field_z)**2)
#     labels.append(f"z = {z:.5f} m")

# intensity_stack = np.asarray(intensity_stack, dtype=np.float32)
# intensity_stack /= intensity_stack.max()

# tifffile.imwrite(
#     'hologram_1obj_reconstruction_stack.tif',
#     intensity_stack,
#     imagej=True,
#     metadata={"axes": "ZYX", "Labels": labels},
# )
# print("Saved hologram_1obj_reconstruction_stack.tif")
# %%
