import numpy as np
import matplotlib.pyplot as plt

# Configuration
filename = '/data-pool/tor/tiny/data/tiny_images.bin'
GRID_SIZE = 100  # 100x100 = 10,000 images
IMG_SIDE = 32
CHANNELS = 3
START_INDEX = 42000 # Starting where you left off

def show_mega_mosaic():
    item_size = IMG_SIDE * IMG_SIDE * CHANNELS
    total_bytes = GRID_SIZE**2 * item_size
    
    # 1. Map the data as a flat byte array first
    data = np.memmap(filename, dtype='uint8', mode='r', 
                     shape=(total_bytes,), 
                     offset=START_INDEX * item_size)

    # 2. Reshape to (Rows, Cols, Channels, Height, Width)
    # Each image is stored as (C, H, W) in the file
    mosaic = data.reshape(GRID_SIZE, GRID_SIZE, CHANNELS, IMG_SIDE, IMG_SIDE)

    # 3. Transpose to (Row, H, Col, W, C) then flatten grid dims
    # Original axes: 0:Row, 1:Col, 2:C, 3:H, 4:W
    # Target order: (0, 3, 1, 4, 2)
    combined = mosaic.transpose(0, 3, 1, 4, 2).reshape(GRID_SIZE * IMG_SIDE,
                                                       GRID_SIZE * IMG_SIDE,
                                                       CHANNELS)

    # 4. Display
    # At 120x32, we have 3840px—exactly half the width of your 8K screen.
    plt.figure(figsize=(16, 16), dpi=100) 
    plt.imshow(combined, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":
    show_mega_mosaic()
