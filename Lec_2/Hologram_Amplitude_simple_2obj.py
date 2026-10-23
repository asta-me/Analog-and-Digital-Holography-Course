#%%
# =============================================================================
# DIGITAL HOLOGRAPHY SIMULATION: TWO OBJECT PLANES
#
# Author: Marco Astarita, PhD
# Date: 2025-10-16
#
# Purpose:
# This script records one hologram of two objects at different depths,
# then reconstructs the field at each real and virtual image plane.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image

#%%
# --- PROPAGATOR DEFINITION ---
def propagate_TF(u_in, dx, lam, z):
    N, M = u_in.shape
    fx = fftshift(fftfreq(M, d=dx))
    fy = fftshift(fftfreq(N, d=dx))
    fX, fY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lam * z * (fX**2 + fY**2))
    U_in = fftshift(fft2(ifftshift(u_in)))
    U_out = U_in * H
    return fftshift(ifft2(ifftshift(U_out)))

#%%
# --- 1. SIMULATION PARAMETERS ---
# Grid Parameters
N = 2048    
M = 2048
dx = 5 * 1e-6
dy = dx

# Physical Parameters
lam = 633 * 1e-9            
k = 2 * np.pi / lam

# Scene Geometry
dist_obj_1 = 0.20  # Distance for object 'M' [m]
dist_obj_2 = 0.25  # Distance for object 'A' [m]

# Holography Parameters (ON-AXIS)
theta_x_deg = 0.0
theta_y_deg = 0.0
balance_factor = 1.5
pupil_radius_factor = 0.9


#%%
# --- 2. OBJECT DEFINITION ---

# Load the pre-generated 2048x2048 object amplitudes.
obj_amp_1 = np.array(Image.open("m_top_left.bmp")) / 255.0
obj_amp_2 = np.array(Image.open("a_bottom_right.bmp")) / 255.0

# Apply random phase to each object to simulate diffuse surfaces
u1 = obj_amp_1 * np.exp(1j * np.random.rand(M, N) * 2 * np.pi)
u2 = obj_amp_2 * np.exp(1j * np.random.rand(M, N) * 2 * np.pi)

#%%
# --- 3. HOLOGRAM RECORDING ---

# Propagate each object wave to the hologram plane
print("Propagating object waves...")
O1 = propagate_TF(u1, dx, lam, dist_obj_1)
O2 = propagate_TF(u2, dx, lam, dist_obj_2)
O = (O1 + O2) * balance_factor

# Define the on-axis reference wave (R)
x_vec = (np.arange(M) - M/2) * dx
y_vec = (np.arange(N) - M/2) * dy
X, Y = np.meshgrid(x_vec, y_vec)

pupil_mask = (np.sqrt(X**2 + Y**2) <= pupil_radius_factor * (M * dx / 2)).astype(float)
R = pupil_mask

# Record the hologram intensity
print("Recording hologram...")
I = np.abs(O + R)**2
hologram = I
# hologram -= np.mean(hologram) # Optional: remove DC background
#%%
# --- 4. HOLOGRAM RECONSTRUCTION ---
R_recon = R
field_after_hologram = R_recon * hologram

# Reconstruct at both object depths, for real and virtual images
print("Reconstructing fields...")
u_real_1 = propagate_TF(field_after_hologram, dx, lam, dist_obj_1)
u_virtual_1 = propagate_TF(field_after_hologram, dx, lam, -dist_obj_1)

u_real_2 = propagate_TF(field_after_hologram, dx, lam, dist_obj_2)
u_virtual_2 = propagate_TF(field_after_hologram, dx, lam, -dist_obj_2)

# Calculate and normalize the reconstructed intensities
I_real_1 = np.abs(u_real_1)**2
I_real_1 /= np.max(I_real_1)

I_virtual_1 = np.abs(u_virtual_1)**2
I_virtual_1 /= np.max(I_virtual_1)

I_real_2 = np.abs(u_real_2)**2
I_real_2 /= np.max(I_real_2)

I_virtual_2 = np.abs(u_virtual_2)**2
I_virtual_2 /= np.max(I_virtual_2)

#%%
# --- 5. VISUALIZATION ---

# Define a normalized clipping value for display
vmax_normalized = 0.75

# Plot the two input objects
plt.figure(figsize=(7, 7))
plt.imshow(np.abs(u1) + np.abs(u2), cmap='gray')
plt.title("Object Amplitudes: M and A")
plt.axis('off')
plt.show()

# Plot the recorded hologram
plt.figure(figsize=(7, 7))
plt.imshow(hologram, cmap='gray')
plt.title("Recorded Hologram")
plt.axis('off')
plt.show()

# Plot the four real and virtual images
plt.figure(figsize=(7, 7))
im1 = plt.imshow(I_real_1, cmap='hot', vmax=vmax_normalized)
plt.title(f"Real Image (Focus @ d={dist_obj_1} m)")
plt.axis('off')
plt.colorbar(im1, label="Normalized Intensity")
plt.show()

# Plot 4: Virtual Image, focused on Object 1 ('M')
plt.figure(figsize=(7, 7))
im2 = plt.imshow(I_virtual_1, cmap='hot', vmax=vmax_normalized)
plt.title(f"Virtual Image (Focus @ d={-dist_obj_1} m)")
plt.axis('off')
plt.colorbar(im2, label="Normalized Intensity")
plt.show()

# Plot 5: Real Image, focused on Object 2 ('A')
plt.figure(figsize=(7, 7))
im3 = plt.imshow(I_real_2, cmap='hot', vmax=vmax_normalized)
plt.title(f"Real Image (Focus @ d={dist_obj_2} m)")
plt.axis('off')
plt.colorbar(im3, label="Normalized Intensity")
plt.show()

# Plot 6: Virtual Image, focused on Object 2 ('A')
plt.figure(figsize=(7, 7))
im4 = plt.imshow(I_virtual_2, cmap='hot', vmax=vmax_normalized)
plt.title(f"Virtual Image (Focus @ d={-dist_obj_2} m)")
plt.axis('off')
plt.colorbar(im4, label="Normalized Intensity")
plt.show()

#%%
# --- 6. OPTIONAL: RECONSTRUCTION STACK FOR IMAGEJ ---
import tifffile

depths = np.linspace(-1.25 * dist_obj_2, 1.25 * dist_obj_2, 41)
intensity_stack = []
labels = []

for z in depths:
    field_z = propagate_TF(field_after_hologram, dx, lam, z)
    intensity_stack.append(np.abs(field_z)**2)
    labels.append(f"z = {z:.5f} m")

intensity_stack = np.asarray(intensity_stack, dtype=np.float32)
intensity_stack /= intensity_stack.max()

tifffile.imwrite(
    'hologram_2obj_reconstruction_stack.tif',
    intensity_stack,
    imagej=True,
    metadata={"axes": "ZYX", "Labels": labels},
)
print("Saved hologram_2obj_reconstruction_stack.tif")
# %%
