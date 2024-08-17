# %% libraries
import matplotlib.pyplot as plt

# %%
def plot_images_grid(images_array, start_N=0, num_images=12, grid_shape=(3, 4)):
    # Create a figure with a specified size
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 10))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for i in range(num_images):
        if i < len(images_array):
            # Plot each image
            axes[i].imshow(images_array[i+start_N])
            axes[i].axis('off')  # Hide the axis

    # Remove any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()