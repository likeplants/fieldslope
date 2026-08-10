import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_random_jp2(folder="output"):
    """
    Loads a random JP2 DEM tile, normalizes it,
    and displays it.
    """

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".jp2")
    ]

    if not files:
        raise RuntimeError(
            "No JP2 files found"
        )

    filename = random.choice(files)

    path = os.path.join(
        folder,
        filename
    )

    print(
        "Loading:",
        path
    )

    data = np.asarray(
        Image.open(path),
        dtype=np.float32
    )

    # normalize for visualization
    img = (
        data - np.min(data)
    ) / (
        np.max(data) - np.min(data)
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        img,
        cmap="terrain"
    )

    plt.colorbar(
        label="normalized elevation"
    )

    plt.title(
        filename
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()


show_random_jp2("srtm_tiles")