import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage import io


def linear_diffusion_convolution(image, num_steps=50, dt=0.1, D=1.0):
    """
    Perform linear diffusion using convolution with the discrete Laplacian.

    Parameters:
        image (np.ndarray): 2D input image
        num_steps (int): number of time steps
        dt (float): time step size
        D (float): diffusion coefficient

    Returns:
        np.ndarray: diffused image
    """
    u = image.astype(np.float64)

    # Discrete Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    for step in range(num_steps):
        delta_u = convolve2d(u, laplacian_kernel, mode='same', boundary='symm')
        u += dt * D * delta_u

    return u

def main():
    image = io.imread('cisco_demo.jpg') # Demo image of my dog

    # If it has 4 channels (RGBA), drop the alpha
    if image.shape[-1] == 4:
        image = image[..., :3]  # Keep only R, G, B

    image = rgb2gray(image) # Grayscale it

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    diffused = linear_diffusion_convolution(image, num_steps=500, dt=0.1, D=0.5)


    plt.subplot(1, 2, 2)
    plt.title('Δt=500')
    plt.imshow(diffused, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()