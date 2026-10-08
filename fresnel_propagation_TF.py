#%%
# =============================================================================
# FRESNEL DIFFRACTION: TRANSFER FUNCTION (TF) METHOD SIMULATION
#
# Author: Marco Astarita
# Date: October 11, 2025
#
# Description:
# This script simulates Fresnel diffraction from a circular aperture using the
# Transfer Function (TF) method.This approach is based on the theory
# described in Chapter 7.5.2 of "Kostuk - Holography Principles and applications".
# For the implementation, I used Chapter 5.1 of "Computational Fourier Optics" by David Voelz,
#
# Sampling validity in x: z <= N*dx**2/lam. Above this, the transfer-function
# phase changes by more than pi between adjacent frequency samples at Nyquist.
# This is a numerical sampling limit; finite-window wrap-around is a separate issue.
#
# To run cells in VS Code, install the Microsoft Jupyter extension.
# Then use "Run Cell" above each cell marker, starting from the top.
# Alternatively, run this file as a regular Python script.
#
# =============================================================================

#%% 0. IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS
N = 1024          # Number of samples along y
M = 1024          # Number of samples along x
dx = 10 * 1e-6    # Sampling interval in the source plane (Δx) [m]
dy = 10 * 1e-6    # Sampling interval in the source plane (Δy) [m]
lam = 633 * 1e-9  # Wavelength (λ) [m]
dist = 0.05       # Propagation distance [m]
max_dist = N * dx**2 / lam  # Maximum valid propagation distance according to sampling criteria

#%% 2. SOURCE PLANE: FIELD DEFINITION AND DISPLAY

# Source plane coordinates (x1, y1)
x1_coords = (np.arange(M) - M // 2) * dx
y1_coords = (np.arange(N) - N // 2) * dy
x1, y1 = np.meshgrid(x1_coords, y1_coords)

# Definition of the source field (input field u1)
aperture_radius = 0.0015  # Aperture radius [m]
r = np.sqrt(x1**2 + y1**2)

# Circular aperture with a hard edge
u1 = (r <= aperture_radius).astype(float)

# Alternative: circular aperture with a smooth edge
# u1 = np.exp(-(r / aperture_radius)**50)

# Alternative: square aperture with hard edges
# u1 = ((np.abs(x1) <= aperture_radius) & \
#       (np.abs(y1) <= aperture_radius)).astype(float)

# Alternative: square aperture with smooth edges
# u1 = np.exp(-((np.abs(x1) / aperture_radius)**50 + \
#               (np.abs(y1) / aperture_radius)**50))

# Display the source field
plt.figure(figsize=(6, 6))
plt.imshow(
    u1,
    cmap='gray',
    extent=[x1_coords[0], x1_coords[-1], y1_coords[0], y1_coords[-1]]
)
plt.title("Input Field (Source Plane)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar()
plt.show()

#%% 3. DESTINATION PLANE

# In the Transfer Function (and Angular Spectrum) method, the destination plane
# has the EXACT SAME coordinates and sampling as the source plane.
x2_coords = x1_coords
y2_coords = y1_coords
x2, y2 = np.meshgrid(x2_coords, y2_coords)

#%% 4. PROPAGATION ALGORITHM

def propagate_Fresnel_TF(u1, dx, dy, lam, z):
    """
    Calculates diffraction using the Fresnel Transfer Function method.
    """
    N, M = u1.shape
    
    # Frequency coordinates (fx, fy) in [1/m]. Often denoted as (u,v) in texts.
    ux_coords = fftshift(fftfreq(M, d=dx))
    uy_coords = fftshift(fftfreq(N, d=dy))
    ux, uy = np.meshgrid(ux_coords, uy_coords)
    
    # Transfer function H for the Fresnel Approximation (from Voelz, Eq. 5.2)
    # Note: The term exp(jkz) is a constant phase offset and can be ignored
    # when only intensity is of interest.
    H = np.exp(-1j * np.pi * lam * z * (ux**2 + uy**2))

    # Propagation in the frequency domain
    U1 = fftshift(fft2(ifftshift(u1)))
    U2 = U1 * H
    u2 = fftshift(ifft2(ifftshift(U2)))
    
    return u2

# Calculate the complex field at the destination plane
u2 = propagate_Fresnel_TF(u1, dx, dy, lam, z=dist)

# Calculation of the intensity (the square of the modulus of the amplitude)
intensity = np.abs(u2)**2

#%% 5. PROPAGATED INTENSITY

# Plot 2: Complete Diffraction Pattern
plt.figure(figsize=(6, 6))
plt.imshow(
    intensity,
    cmap='gray',
    extent=[x2_coords[0], x2_coords[-1], y2_coords[0], y2_coords[-1]]
)
plt.title(f"Diffraction Pattern at d = {dist} m (Full)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar(label="Intensity")
plt.show()

# Plot 3: Zoom on the Central Diffraction Pattern
plt.figure(figsize=(6, 6))
plt.imshow(
    intensity,
    cmap='gray',
    extent=[x2_coords[0], x2_coords[-1], y2_coords[0], y2_coords[-1]]
)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Diffraction Pattern at d = {dist} m (Zoom)")
# Axis limits to display the central area
zoom_limit = 1.5e-3
plt.xlim(-zoom_limit, zoom_limit)
plt.ylim(-zoom_limit, zoom_limit)
plt.colorbar(label="Intensity")
plt.show()

#%% 6. CENTRAL-ROW PROFILE
# Extraction of the central row from the intensity matrix
central_row_index = N // 2
central_row_intensity = intensity[central_row_index, :]

# Plot 4: 1D profile of the central row with the same zoom
plt.figure(figsize=(8, 4))
plt.plot(x2_coords, central_row_intensity)
plt.title(f"Intensity Profile of the Central Row (d = {dist} m)")
plt.xlabel("x [m]")
plt.ylabel("Intensity")
plt.grid(True)
# Apply the same x-axis limits as the zoomed plot
plt.xlim(-zoom_limit, zoom_limit)
plt.show()

#%% 7. SAVE PROPAGATION STACKS FOR IMAGEJ
depths = np.linspace(0 , 0.1, 41)
intensity_stack = []
central_profiles = []
labels = []

for z in depths:
    field_z = propagate_Fresnel_TF(u1, dx, dy, lam, z)
    intensity_z = np.abs(field_z)**2
    intensity_stack.append(intensity_z)
    central_profiles.append(intensity_z[N // 2, :])
    labels.append(f"z = {z:.5f} m")

intensity_stack = np.asarray(intensity_stack, dtype=np.float32)
central_profiles = np.asarray(central_profiles, dtype=np.float32)
intensity_scale = np.max(intensity_stack)
intensity_stack /= intensity_scale
central_profiles /= intensity_scale

profile_plot_stack = []
for label, profile_z in zip(labels, central_profiles):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(x2_coords, profile_z)
    ax.set_title(f"Central-row intensity profile, {label}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Normalized intensity")
    ax.set_xlim(-zoom_limit, zoom_limit)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    profile_plot_stack.append(
        np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2).astype(np.uint8)
    )
    plt.close(fig)

tifffile.imwrite(
    "fresnel_TF_intensity_stack.tif",
    intensity_stack,
    imagej=True,
    metadata={"labels": labels}
)
tifffile.imwrite(
    "fresnel_TF_central_profiles.tif",
    np.asarray(profile_plot_stack, dtype=np.uint8),
    imagej=True,
    metadata={"labels": labels}
)
print("Saved fresnel_TF_intensity_stack.tif and fresnel_TF_central_profiles.tif")
# %%
