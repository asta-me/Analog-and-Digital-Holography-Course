#%%
# =============================================================================
# FAR-FIELD (FRAUNHOFER) DIFFRACTION SIMULATION
#
# Author: Marco Astarita
# Date: October 11, 2025
#
# Description:
# This script simulates the Far-Field (Fraunhofer) diffraction from an aperture
# using the single Fourier transform method. This approach is based on the theory
# described in Chapter 2.6.1 of "Kostuk - Holography Principles and applications"

# To run cells in VS Code, install the Microsoft Jupyter extension.
# Then use "Run Cell" above each cell marker, starting from the top.
# Alternatively, run this file as a regular Python script.
#
# =============================================================================

#%% 0. IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

#%% 1. PHYSICAL PARAMETERS
N = 2*1024          # Number of samples along y
M = 2*1024          # Number of samples along x
dx = 10 * 1e-6    # Sampling interval in the source plane (Δx) [m]
dy = 10 * 1e-6    # Sampling interval in the source plane (Δy) [m]
lam = 633 * 1e-9  # Wavelength (λ) [m]
dist = 10.0       # Propagation distance [m]

# Note: Fraunhofer condition d >> 2*D^2/λ.
# For D=1mm (aperture diameter), d >> 3.16m. Our distance satisfies this.

#%% 2. SOURCE PLANE: FIELD DEFINITION AND DISPLAY

# Source plane coordinates (x1, y1)
x1_coords = (np.arange(M) - M // 2) * dx
y1_coords = (np.arange(N) - N // 2) * dy
x1, y1 = np.meshgrid(x1_coords, y1_coords)

# Definition of the source field (input field u1)
aperture_radius = 0.0005  # Aperture radius [m] D = 1 mm
r = np.sqrt(x1**2 + y1**2)

# Circular aperture with a hard edge
u1 = (r <= aperture_radius).astype(float)
# Alternative: circular aperture with a smooth edge
# u1 = np.exp(-(r / aperture_radius)**50)
# Alternative: square aperture with hard edges
# u1 = ((np.abs(x1) <= aperture_radius) & \
#       (np.abs(y1) <= aperture_radius)).astype(float)
# Alternative: square aperture with smooth edges
u1 = np.exp(-((np.abs(x1) / aperture_radius)**50 + \
              (np.abs(y1) / aperture_radius)**50))

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

# Scaling the frequency coordinates to get spatial coordinates
x2_coords = lam * dist * ux
y2_coords = lam * dist * uy
x2, y2 = np.meshgrid(x2_coords, y2_coords)

#%% 4. PROPAGATION ALGORITHM

def fraunhofer_far_field(u1, lam, z):
    """
    Calculates Fraunhofer diffraction using the single-step FFT method.
    """
    # In far-field, the FFT is applied directly to the source field.
    u2 = fftshift(fft2(ifftshift(u1)))
    # Constant factors have been neglected for simplicity.
    return u2

# Calculate the complex field at the destination plane
u2 = fraunhofer_far_field(u1, lam, z=dist)

# Calculation of the intensity (the square of the modulus of the amplitude)
intensity = np.abs(u2)**2

# Optional contrast enhancement to make the faint rings more visible
gamma = 1
intensity_display = intensity**gamma

#%% 5. PROPAGATED INTENSITY

# Complete Diffraction Pattern
plt.figure()
plt.imshow(
    intensity_display,
    cmap='gray',
    extent=[x2_coords[0], x2_coords[-1], y2_coords[0], y2_coords[-1]]
)
plt.title(f"Far-Field Diffraction Pattern at d = {dist} m (Full)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar(label="Intensity (contrast enhanced)")
plt.show()

# Zoom on the Central Diffraction Pattern
plt.figure()
plt.imshow(
    intensity_display,
    cmap='gray',
    extent=[x2_coords[0], x2_coords[-1], y2_coords[0], y2_coords[-1]]
)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Far-Field Diffraction Pattern at d = {dist} m (Zoom)")
# Axis limits to display the central area
zoom_limit = 5e-2
plt.xlim(-zoom_limit, zoom_limit)
plt.ylim(-zoom_limit, zoom_limit)
plt.colorbar(label=f"Intensity (contrast enhanced gamma={gamma})")
plt.show()


# Zoom on the Central Diffraction Pattern + Saturation
# Axis limits to display the central area
zoom_limit = 5e-2
# Saturation threshold
saturation_threshold = 1/50

plt.figure(figsize=(24, 12))
plt.imshow(
    intensity,
    cmap='gray',
    extent=[x2_coords[0], x2_coords[-1], y2_coords[0], y2_coords[-1]],
    vmin=0, vmax=saturation_threshold * intensity.max()
)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Far-Field Diffraction Pattern at d = {dist} m (Zoom)")
plt.xlim(-zoom_limit, zoom_limit)
plt.ylim(-zoom_limit, zoom_limit)
plt.colorbar(label=f"Intensity (Saturated at {saturation_threshold})")
plt.show()



#%% 6. CENTRAL-ROW PROFILE
# Extraction of the central row from the intensity matrix
central_row_index = N // 2
central_row_intensity = intensity_display[central_row_index, :]

# Plot 4: 1D profile of the central row with the same zoom
plt.figure(figsize=(8, 4))
plt.plot(x2_coords, central_row_intensity)
plt.title(f"Intensity Profile of the Central Row (d = {dist} m)")
plt.xlabel("x [m]")
plt.ylabel("Intensity (contrast enhanced)")
plt.grid(True)
# Apply the same x-axis limits as the zoomed plot
plt.xlim(-zoom_limit, zoom_limit)
plt.show()

