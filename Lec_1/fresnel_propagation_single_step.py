#%%
# =============================================================================
# FRESNEL DIFFRACTION SIMULATION (SINGLE-STEP FFT METHOD)
#
# Author: [Marco Astarita]
# Date: October 08, 2025
#
# Description:
# This script simulates the Fresnel diffraction from an aperture using
# the "single-step Fresnel Propagation" method.
# This approach is based on the theory described in Chapter 2.6.1 of "Kostuk - Holography Principles and applications"
# For the implementation, I used the Chapter 6.3 of "Introduction to Computer Holography" by Kyoji Matsushima.

# Sampling validity in x: z >= 2*a_x*dx/lam, where a_x is the largest |x|
# at which the input field is appreciable. Below this, the input quadratic
# phase changes by more than pi per sample and is undersampled.
# This is a numerical sampling limit, not a physical limit on propagation.
#
# The destination plane coordinates are explicitly calculated from the
# spatial frequencies of the source grid, as derived from the FFT relationship.
# To run cells in VS Code, install the Microsoft Jupyter extension.
# Then use "Run Cell" above each cell marker, starting from the top.
# Alternatively, run this file as a regular Python script.
# =============================================================================

#%% 0. IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS
N = 1024          # Number of samples along y
M = 1024          # Number of samples along x
dx = 10 * 1e-6    # Sampling interval in the source plane (Δxs) [m]
dy = 10 * 1e-6    # Sampling interval in the source plane (Δys) [m]
lam = 633 * 1e-9  # Wavelength (λ) [m]
dist = 0.5        # Propagation distance [m] (0.16 to 0.4 works fine)

#%% 2. SOURCE PLANE: FIELD DEFINITION AND DISPLAY

# Source plane coordinates (x1, y1)
x1_coords = (np.arange(M) - M // 2) * dx
y1_coords = (np.arange(N) - N // 2) * dy
x1, y1 = np.meshgrid(x1_coords, y1_coords)

# Definition of the source field (input field u1)
aperture_radius = 0.0015  # Aperture radius [m]
min_dist = 2 * aperture_radius * dx / lam  # Minimum valid propagation distance according to sampling criteria
r = np.sqrt(x1**2 + y1**2)

# Circular aperture with a smooth edge
u1 = np.exp(-((x1**2 + y1**2)**0.5 / aperture_radius)**50)

# Alternative: circular aperture with a hard edge
# u1 = (r <= aperture_radius).astype(float)

# Alternative: square aperture with hard edges
# u1 = ((np.abs(x1) <= aperture_radius) & \
#       (np.abs(y1) <= aperture_radius)).astype(float)

# Alternative: square aperture with smooth edges
# u1 = np.exp(-((np.abs(x1) / aperture_radius)**50 + \
#               (np.abs(y1) / aperture_radius)**50))

# Display the source field
plt.figure()
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

# The destination plane coordinates (x2, y2) scale with distance
# according to the Fourier transform properties of diffraction.

# Frequencies of the fft transform are given by
ux = fftshift(fftfreq(M, dx))
uy = fftshift(fftfreq(N, dy))
# Equivalent to going from the min to the max of Nyquist frequency in n steps
# ux = np.linspace(-1, 1, M, endpoint=False) * (1 / (2 * dx))
# uy = np.linspace(-1, 1, N, endpoint=False) * (1 / (2 * dy))

# Scaling the frequency coordinates to get spatial coordinates
x2_coords = lam * dist * ux
y2_coords = lam * dist * uy
x2, y2 = np.meshgrid(x2_coords, y2_coords)

#%% 4. PROPAGATION ALGORITHM

def fresnel_single_step(u1, x1, y1, lam, z):
    """
    Calculates Fresnel diffraction using the single-step FFT method.
    """
    # Multiplication by the pre-FFT quadratic phase factor
    u1_ = u1 * np.exp(1j * (np.pi / (lam * z)) * (x1**2 + y1**2))
    
    # Execution of the 2D Fourier Transform
    u2_ = fftshift(fft2(ifftshift(u1_)))

    # For intensity calculation, the phase terms are not needed, so a simpler scaling is often used.
    u2 = u2_ / (lam * z)

    return u2

# Calculate the complex field at the destination plane
u2 = fresnel_single_step(u1, x1, y1, lam, z=dist)

# Scaling Factor to get the final COMPLEX FIELD
# u2 *= np.exp(1j * (2*np.pi/lam) * z) \
#      * np.exp(1j * (np.pi / (lam * z)) * (x2**2 + y2**2))    

# Calculation of the intensity (the square of the modulus of the amplitude)
intensity = np.abs(u2)**2

#%% 5. PROPAGATED INTENSITY
# Plot 2: Complete Diffraction Pattern
plt.figure()
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
plt.figure()
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
depths = np.linspace(0.05 , 0.5, 41)
intensity_stack = []
central_profiles = []
labels = []

for z in depths:
    field_z = fresnel_single_step(u1, x1, y1, lam, z)
    intensity_z = np.abs(field_z)**2
    intensity_stack.append(intensity_z)
    central_profiles.append(intensity_z[N // 2, :])

    dx_out = lam * z / (M * dx)
    dy_out = lam * z / (N * dy)
    labels.append(f"z = {z:.5f} m; dx = {dx_out:.3e} m; dy = {dy_out:.3e} m")

intensity_stack = np.asarray(intensity_stack, dtype=np.float32)
central_profiles = np.asarray(central_profiles, dtype=np.float32)
intensity_scale = np.max(intensity_stack)
intensity_stack /= intensity_scale
central_profiles /= intensity_scale

profile_plot_stack = []
for z, label, profile_z in zip(depths, labels, central_profiles):
    x_coords_z = lam * z * fftshift(fftfreq(M, dx))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(x_coords_z, profile_z)
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
    "fresnel_single_step_intensity_stack.tif",
    intensity_stack,
    imagej=True,
    metadata={"labels": labels}
)
tifffile.imwrite(
    "fresnel_single_step_central_profiles.tif",
    np.asarray(profile_plot_stack, dtype=np.uint8),
    imagej=True,
    metadata={"labels": labels}
)
print("Saved fresnel_single_step_intensity_stack.tif and fresnel_single_step_central_profiles.tif")
# %%
