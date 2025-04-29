import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def total_variation_flow_central(u, num_iter=100, dt=0.1, epsilon=1e-10):
    u = u.copy()

    for _ in range(num_iter):
        # Central differences
        ux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / 2
        uy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / 2

        # Gradient magnitude with regularization
        grad_mag = np.sqrt(ux**2 + uy**2 + epsilon**2)

        # Normalized gradients
        nx = ux / grad_mag
        ny = uy / grad_mag

        # Divergence (central difference of normalized gradients)
        div_nx = (np.roll(nx, -1, axis=1) - np.roll(nx, 1, axis=1)) / 2
        div_ny = (np.roll(ny, -1, axis=0) - np.roll(ny, 1, axis=0)) / 2
        div = div_nx + div_ny

        # Update step (explicit Euler)
        u += dt * div

    return u

# Example usage
# Load an image
image = io.imread('cisco_demo.jpg')
if image.ndim == 3:
    image = color.rgb2gray(image)

# Normalize image
image = image / np.max(image)

# Apply TV flow (central differences version)
u_tv_central = total_variation_flow_central(image, num_iter=200, dt=0.1)

# Plotting
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Δt=200")
plt.imshow(u_tv_central, cmap='gray')
plt.axis('off')
plt.show()