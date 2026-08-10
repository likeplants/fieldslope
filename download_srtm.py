import os
import re
import json
import shutil
import zipfile
import requests

import numpy as np
from PIL import Image


BASE_URL = "https://step.esa.int/auxdata/dem/SRTMGL1/"

OUTPUT_DIR = "srtm_tiles"
TEMP_DIR = "temp"
METADATA_FILE = "metadata.json"


def get_tiles():
    """Scrape available SRTM tiles from ESA directory listing."""

    print("Scraping tile list...")

    response = requests.get(
        BASE_URL,
        timeout=30
    )

    response.raise_for_status()

    tiles = re.findall(
        r'href="([^"]+\.SRTMGL1\.hgt\.zip)"',
        response.text
    )

    tiles = sorted(set(tiles))

    print(
        f"Found {len(tiles)} tiles"
    )

    return tiles



def clear_temp():

    os.makedirs(
        TEMP_DIR,
        exist_ok=True
    )

    for item in os.listdir(TEMP_DIR):

        path = os.path.join(
            TEMP_DIR,
            item
        )

        if os.path.isdir(path):
            shutil.rmtree(path)

        else:
            os.remove(path)



def download_tile(filename):

    url = BASE_URL + filename

    local_zip = os.path.join(
        TEMP_DIR,
        filename
    )

    print(
        f"Downloading {filename}"
    )

    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    with open(
        local_zip,
        "wb"
    ) as f:
        f.write(
            response.content
        )

    return local_zip



def extract_hgt(zip_path):

    print(
        "Extracting..."
    )

    with zipfile.ZipFile(zip_path) as z:

        z.extractall(
            TEMP_DIR
        )

        hgt_files = [
            f for f in z.namelist()
            if f.lower().endswith(".hgt")
        ]

    if not hgt_files:
        raise RuntimeError(
            "No HGT file found in archive"
        )

    return os.path.join(
        TEMP_DIR,
        hgt_files[0]
    )



def read_hgt(path):

    print(
        "Reading elevation data..."
    )

    data = np.fromfile(
        path,
        dtype=">i2"
    )

    size = int(
        np.sqrt(
            len(data)
        )
    )

    if size * size != len(data):
        raise RuntimeError(
            "Invalid HGT dimensions"
        )

    return data.reshape(
        size,
        size
    )



def save_jpeg2000(data, tile):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    data = data.astype(
        np.int16
    )

    # SRTM void values
    data[data <= -32768] = 0


    print(
        f"Elevation range: {data.min()} - {data.max()}"
    )


    image = Image.fromarray(
        data,
        mode="I;16"
    )


    output_name = tile.replace(
        ".hgt.zip",
        ".jp2"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    print(
        f"Saving {output_path}"
    )


    image.save(
        output_path,
        format="JPEG2000",
        irreversible=True,
        quality_mode="rates",
        quality_layers=[10]
    )


def load_metadata():

    if os.path.exists(
        METADATA_FILE
    ):

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return {}



def save_metadata(metadata):

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )



def process_tile(tile):

    jpeg_name = tile.replace(
        ".hgt.zip",
        ".jp2"
    )

    jpeg_path = os.path.join(
        OUTPUT_DIR,
        jpeg_name
    )


    if os.path.exists(
        jpeg_path
    ):

        print(
            f"Skipping existing {jpeg_name}"
        )

        return None


    clear_temp()


    zip_path = download_tile(
        tile
    )

    hgt_path = extract_hgt(
        zip_path
    )

    elevation = read_hgt(
        hgt_path
    )


    save_jpeg2000(
        elevation,
        tile
    )


    return {
        "tile": tile,
        "jpeg": jpeg_name,
        "shape": list(
            elevation.shape
        ),
        "min_elevation": int(
            np.nanmin(elevation)
        ),
        "max_elevation": int(
            np.nanmax(elevation)
        )
    }



def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    metadata = load_metadata()


    tiles = get_tiles()


    for index, tile in enumerate(
        tiles,
        1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"[{index}/{len(tiles)}] {tile}"
        )


        try:

            result = process_tile(
                tile
            )


            if result:

                metadata[tile] = result


                # IMPORTANT:
                # save immediately after each tile
                save_metadata(
                    metadata
                )

                print(
                    "Metadata saved."
                )


        except Exception as e:

            print(
                f"FAILED {tile}: {e}"
            )


    print(
        "\nFinished."
    )



if __name__ == "__main__":
    main()