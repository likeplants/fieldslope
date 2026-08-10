import requests
from io import BytesIO
from PIL import Image
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import tqdm

from get_tiles import osm_tile_bounds, srtm_where, read_osm_dem_tile, read_osm_dem_tile
from rendering import *

def render_tile_comparison(
        x,
        y,
        z,
        axes,
        srtm_tiles="./srtm_tiles"
):
    """
    Render OSM, contour and terrain views for one XYZ tile
    into an existing matplotlib axes row.
    """

    # -----------------------------
    # OSM image
    # -----------------------------

    url = (
        f"https://a.basemaps.cartocdn.com/"
        f"rastertiles/voyager/"
        f"{z}/{x}/{y}@2x.png"
    )

    headers = {
        "User-Agent": (
            "SRTM-Contour-Renderer/1.0 "
            "(contact@example.com)"
        ),
        "Accept": "image/*",
        "Referer": "http://localhost:3000/",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    osm_img = Image.open(
        BytesIO(response.content)
    )


    # -----------------------------
    # DEM extraction
    # -----------------------------

    bounds = osm_tile_bounds(
        x=x,
        y=y,
        z=z
    )

    dem = read_osm_dem_tile(
        bounds,
        srtm_tiles
    )


    # -----------------------------
    # Resize DEM to tile resolution
    # -----------------------------

    target_size = osm_img.size[0]   # 512 for @2x tiles

    sy = target_size / dem.shape[0]
    sx = target_size / dem.shape[1]

    dem = zoom(
        dem,
        (sy, sx),
        order=1
    )


    # -----------------------------
    # Rendering parameters
    # -----------------------------

    contour_step, smoothing = contour_settings_for_zoom(
        z
    )

    dem = np.nan_to_num(
        dem,
        nan=np.nanmin(dem))#replaces nan

    contour_img = render_heightmap(
        dem,
        method="contour",
        contour_step=contour_step,
        smoothing=smoothing
    )


    terrain_img = render_heightmap(
        dem,
        method="cmap",
        smoothing=smoothing
    )


    # -----------------------------
    # Plot into supplied axes
    # -----------------------------

    axes[0].imshow(
        osm_img
    )
    axes[0].set_title(
        f"OSM z={z} x={x} y={y}"
    )
    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        contour_img,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[1].set_title(
        f"Contours {contour_step} m"
    )
    axes[1].axis(
        "off"
    )


    axes[2].imshow(
        terrain_img
    )
    axes[2].set_title(
        "Terrain"
    )
    axes[2].axis(
        "off"
    )


    return {
        "bounds": bounds,
        "dem": dem,
        "contour_step": contour_step,
        "smoothing": smoothing
    }

def render_hillshade_contour_patch(
    center_tile,
    nx=5,
    ny=5,
    srtm_tiles="./srtm_tiles",
    load_osm=False,
):
    """
    Render a patch of hillshade + contours around a center OSM tile.

    Parameters
    ----------
    center_tile : tuple
        (x, y, z) center tile.

    nx, ny : int
        Number of tiles horizontally and vertically.

    srtm_tiles : str
        DEM tile folder.

    load_osm : bool
        Load and return OpenTopoMap tiles.
    """

    cx, cy, z = center_tile

    hx = nx // 2
    hy = ny // 2

    tiles = [
        (x, y, z)
        for y in range(cy - hy, cy + hy + 1)
        for x in range(cx - hx, cx + hx + 1)
    ]

    dems = {}

    if load_osm:
        osms = {}

        headers = {
            "User-Agent": (
                "SRTM-Contour-Renderer/1.0 "
                "(contact@example.com)"
            ),
            "Accept": "image/*",
        }


    for tile in tqdm(tiles, "loading"):

        x, y, z = tile

        if load_osm:

            url = (
                f"https://a.tile.opentopomap.org/"
                f"{z}/{x}/{y}.png"
            )

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=10
                )

                response.raise_for_status()

                osm_img = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

            except Exception:
                osm_img = Image.new(
                    "RGB",
                    (256, 256)
                )

            osms[tile] = osm_img


        bounds = osm_tile_bounds(
            x,
            y,
            z
        )

        try:

            dem = read_osm_dem_tile(
                bounds,
                srtm_tiles=srtm_tiles
            )

            sy = 512 / dem.shape[0]
            sx = 512 / dem.shape[1]

            dem = zoom(
                dem,
                (sy, sx),
                order=1
            )

        except Exception as e:
            print(e)
            dem = np.zeros(
                (512, 512),
                dtype=np.float32
            )

        dems[tile] = dem


    contour_step, smoothing = contour_settings_for_zoom(z)

    rendered = {}

    for tile, heightmap in tqdm(
        dems.items(),
        "Rendering"
    ):
        rendered[tile] = hillshade_contours(
            heightmap,
            contour_step=contour_step,
            sigma=smoothing,
        )


    rows = []

    for y in range(cy - hy, cy + hy + 1):

        row = []

        for x in range(cx - hx, cx + hx + 1):
            row.append(
                rendered[(x, y, z)]
            )

        rows.append(
            np.hstack(row)
        )

    patch = np.vstack(rows)


    if not load_osm:
        return patch


    rows = []

    for y in range(cy - hy, cy + hy + 1):

        row = []

        for x in range(cx - hx, cx + hx + 1):

            row.append(
                np.array(
                    osms[(x, y, z)]
                )
            )

        rows.append(
            np.hstack(row)
        )

    osm_patch = np.vstack(rows)

    return patch, osm_patch


def main():

    if False:

        tiles = [
            # Northern Italy
            (134, 92, 8),

            # Wiesloch: example zoom 13 
            (4294, 2803, 13),

            # Heidelberg: example zoom 12
            (2146, 1399, 12),
        ]

        fig, axes = plt.subplots(
            len(tiles),
            3,
            figsize=(18, 6 * len(tiles))
        )

        for row, (x, y, z) in enumerate(tiles):

            render_tile_comparison(
                x=x,
                y=y,
                z=z,
                axes=axes[row],
                srtm_tiles="./strm_tiles"
            )

        plt.tight_layout()
        plt.show()

    if True:
        GET_OSM = False

        center_tiles = {
            "geneva": {
                9: (260, 178, 9),
            },
            "heidelberg": {
                12: (2146, 1399, 12),
                11: (1073, 699, 11),
                10: (536, 349, 10),
                9: (268, 174, 9),
            }
        }

        for center_tile in center_tiles["heidelberg"].values():

            result = render_hillshade_contour_patch(
                center_tile=center_tile,
                nx=1,
                ny=1,
                load_osm=GET_OSM,
            )

            if GET_OSM:
                patch, osm_patch = result
            else:
                patch = result


            fig, ax = plt.subplots(
                2 if GET_OSM else 1,
                1,
                figsize=(14, 7),
                constrained_layout=True
            )

            if not GET_OSM:
                ax = [ax]


            ax[0].imshow(
                patch,
                cmap="gray",
                vmin=0,
                vmax=255
            )
            ax[0].set_title(
                "Generated hillshade + contours"
            )
            ax[0].axis("off")


            if GET_OSM:

                ax[1].imshow(
                    osm_patch
                )
                ax[1].set_title(
                    "OpenTopoMap"
                )
                ax[1].axis("off")


            for a in ax:
                img = a.images[0].get_array()

                for x in range(0, img.shape[1], 512):
                    a.axvline(
                        x,
                        linewidth=0.8
                    )

                for y in range(0, img.shape[0], 512):
                    a.axhline(
                        y,
                        linewidth=0.8
                    )


            plt.show()

if __name__ == "__main__":
    main()