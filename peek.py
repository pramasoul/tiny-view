import numpy as np
import matplotlib.pyplot as plt

# Path to your 228G file
filename = '/data-pool/tor/tiny/data/tiny_images.bin'
num_images = 79302017

# Memory-map the file without loading it into RAM
# We use shape (N, 3, 32, 32) because the data is packed as RRR...GGG...BBB...
data = np.memmap(filename, dtype='uint8', mode='r', shape=(num_images, 3, 32, 32))

def show_tiny_image(index):
    # Extract the image and transpose from (C, H, W) to (H, W, C) for plotting
    img = data[index].transpose(1, 2, 0)
    
    plt.imshow(img)
    plt.title(f"Tiny Image Index: {index}")
    plt.axis('off')
    plt.show()

# Jump to a random image instantly
show_tiny_image(42000)
