import os
import zipfile
import tempfile
import requests
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import random
import json
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt



BASE_URL = "https://step.esa.int/auxdata/dem/SRTMGL1/"


def load_metadata(path="metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_original_hgt(tile):
    """
    Downloads one SRTM tile, extracts it into a temporary directory,
    loads the HGT as a numpy array and removes all temporary files.

    Parameters
    ----------
    tile : str
        Example:
            N49E008.SRTMGL1.hgt.zip

    Returns
    -------
    np.ndarray
        float32 elevation array
    """

    url = BASE_URL + tile

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = os.path.join(
            temp_dir,
            tile
        )

        print(f"Downloading {tile}...")

        r = requests.get(url, timeout=120)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(temp_dir)

        hgt_file = None

        for file in os.listdir(temp_dir):
            if file.lower().endswith(".hgt"):
                hgt_file = os.path.join(temp_dir, file)
                break

        if hgt_file is None:
            raise RuntimeError("No HGT file found.")

        data = np.fromfile(
            hgt_file,
            dtype=">i2"
        )

        size = int(np.sqrt(len(data)))

        data = data.reshape(
            size,
            size
        )

        return data.astype(np.float32)


def load_srtm_jp2(folder, metadata):

    filename = os.path.join(
        folder,
        metadata["jpeg"]
    )

    data = np.asarray(
        Image.open(filename),
        dtype=np.float32
    )

    # SRTM NoData
    data[data >= 10000] = np.nan

    return data


def analyse(original, reconstructed):
    """
    Computes reconstruction metrics and plots:

    Figure 1:
        Histogram of reconstructed-original

    Figure 2:
        Center row comparison

    Statistics:
        Computed for Gaussian smoothing sigma = 0, 1, 5
        where both original and reconstructed are smoothed
        before comparison.
    """


    print()
    print("Statistics")
    print("=" * 60)


    for sigma in [0, 1, 5]:
        original = np.nan_to_num(
            original,
            nan=0
        )

        reconstructed = np.nan_to_num(
            reconstructed,
            nan=0
        )

        if sigma > 0:

            original_s = gaussian_filter(
                original,
                sigma=sigma
            )

            reconstructed_s = gaussian_filter(
                reconstructed,
                sigma=sigma
            )

        else:

            original_s = original
            reconstructed_s = reconstructed


        diff_s = reconstructed_s - original_s


        mse = np.mean(
            diff_s ** 2
        )

        rmse = np.sqrt(
            mse
        )

        mae = np.mean(
            np.abs(diff_s)
        )

        max_error = np.max(
            np.abs(diff_s)
        )

        bias = np.mean(
            diff_s
        )

        correlation = np.corrcoef(
            original_s.ravel(),
            reconstructed_s.ravel()
        )[0, 1]


        print()
        print(f"Smoothing sigma = {sigma}")
        print("-" * 40)
        print(f"MSE         : {mse:.6f}")
        print(f"RMSE        : {rmse:.6f}")
        print(f"MAE         : {mae:.6f}")
        print(f"Bias        : {bias:.6f}")
        print(f"Max Error   : {max_error:.6f}")
        print(f"Correlation : {correlation:.8f}")

    return
    # --------------------------------------------------
    # Plots: always use original unsmoothed data
    # --------------------------------------------------

    diff = reconstructed - original


    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        diff.ravel(),
        bins=100
    )

    plt.xlabel(
        "Reconstructed - Original [m]"
    )

    plt.ylabel(
        "Pixels"
    )

    plt.title(
        "Elevation reconstruction error"
    )

    plt.tight_layout()


    center = original.shape[0] // 2


    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        original[center],
        label="Original HGT",
        linewidth=1
    )

    plt.plot(
        reconstructed[center],
        label="Reconstructed JPEG",
        linewidth=1
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Elevation [m]"
    )

    plt.title(
        "Center Row"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

def select_random_tile(output_folder, metadata_path):
    """
    Selects a random JPEG from output folder and returns
    the corresponding tile metadata.
    """

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)


    jpg_files = [
        f for f in os.listdir(output_folder)
        if f.lower().endswith(".jp2")
    ]

    jpg = random.choice(
        jpg_files
    )

    for tile, info in metadata.items():

        if info["jpeg"] == jpg:
            return tile, info

    raise RuntimeError(
        "Metadata entry not found"
    )

if True:
    metadata = load_metadata()

    tile = "N49E008.SRTMGL1.hgt.zip"

    original = load_original_hgt(tile)

    reconstructed = load_srtm_jp2(
        "srtm_tiles",
        metadata[tile]
    )
else:
    output_folder = "srtm_tiles"

    tile, tile_metadata = select_random_tile(
        output_folder,
        "metadata.json"
    )

    original = load_original_hgt(
    tile)

    reconstructed = load_srtm_jp2(
        output_folder,
        tile_metadata
    )


    bias = 0
    reconstructed = reconstructed + bias

analyse(
    original,
    reconstructed
)