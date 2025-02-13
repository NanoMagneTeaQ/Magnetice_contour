import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# New dataset
x_new = np.array([3.616, 3.184, 2.968, 2.916, 2.61, 2.358, 2.262, 2.008, 1.714, 1.286, 0.858, 0.43, 0.026,0])
Hx_new = np.array([440, 443, 450, 454, 459, 463, 464, 465, 466, 467, 467, 468, 468, 470])
y_new = np.array([3.624, 3.272, 3.178, 3.114, 2.898, 2.734, 2.574, 2.412, 2.252, 2.09, 1.93, 1.768, 1.608, 
                  1.446, 1.286, 1.124, 0.964, 0.802, 0.642, 0.48, 0.32, 0.158,0])
Hy_new = np.array([438, 444, 452, 454, 458, 463, 464, 464, 465, 465, 465, 465, 465, 
                   466, 466, 466, 466, 466, 467, 467, 468, 468, 470])

# Combine data into radial coordinates
points_new = np.array([[xi, 0] for xi in x_new] + [[0, yi] for yi in y_new])
values_new = np.concatenate((Hx_new, Hy_new))

# Calculate radial distances from the center (0, 0)
radii = np.linalg.norm(points_new, axis=1)

# Handle duplicate radii by averaging field values
unique_radii, indices = np.unique(radii, return_inverse=True)
avg_values = np.zeros_like(unique_radii)
for i, r in enumerate(unique_radii):
    avg_values[i] = np.mean(values_new[indices == i])

# Generate a grid of x, y values
grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
grid_r = np.sqrt(grid_x**2 + grid_y**2)  # Radial distances for the grid

# Interpolate the magnetic field values as a function of radius
grid_H = griddata(unique_radii, avg_values, grid_r.flatten(), method='linear').reshape(grid_r.shape)

# Correct the center point (0, 0) to 470
grid_H[grid_r == 0] = 470

# Mask values outside the maximum radius
grid_H[grid_r > max(unique_radii)] = np.nan

# Apply Gaussian smoothing to blend the center value smoothly
grid_H = gaussian_filter(grid_H, sigma=2)

# Plot circularly symmetric contour with color gradient
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_H, levels=50, cmap='viridis')
plt.colorbar(label='Magnetic Field (H)')
plt.title('Corrected Circularly Symmetric Contour Plot (Color Gradient)')
plt.xlabel('Distance (x-axis)')
plt.ylabel('Distance (y-axis)')
plt.axis('equal')
plt.show()
# plt.savefig(f"/<FILE PATH>", bbox_inches='tight', pad_inches=0.1)  # This will save the image in the current working directory
# plt.close()

# Plot circularly symmetric contour without color gradient
plt.figure(figsize=(10, 8))
contours = plt.contour(grid_x, grid_y, grid_H, levels=20, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.title('Corrected Circularly Symmetric Contour Plot')
plt.xlabel('Distance (x-axis)')
plt.ylabel('Distance (y-axis)')
plt.axis('equal')
plt.show()
# plt.savefig(f"<FILE PATH>", bbox_inches='tight', pad_inches=0.1)  # This will save the image in the current working directory
# plt.close()
