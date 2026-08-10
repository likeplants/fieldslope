import math
import os
import rasterio
import numpy as np
from rasterio.windows import Window
from scipy.ndimage import zoom
import os
import rasterio
import os
import uuid
ENABLE_RASTER_CACHE = True

def open_cached_raster(
    filename,
    cache_dir="./raster_cache",
    max_cache_mb = 1000,
):
    """
    Open raster through a decoded GeoTIFF cache.

    Returns an open rasterio dataset.
    """

    if not ENABLE_RASTER_CACHE:
        return rasterio.open(filename)
    
    def write_atomic(filename, data):
        tmp = f"{filename}.{uuid.uuid4()}.tmp"

        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": data.dtype,
            "compress": None,
        }

        try:
            with rasterio.open(
                tmp,
                "w",
                **profile
            ) as dst:
                dst.write(
                    data,
                    1
                )

            os.replace(
                tmp,
                filename
            )

        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    os.makedirs(cache_dir, exist_ok=True)

    basename = os.path.basename(filename)
    cached = os.path.join(
        cache_dir,
        os.path.splitext(basename)[0] + ".tif"
    )

    # Use cache if available
    if os.path.exists(cached):
        try:
            return rasterio.open(cached)
        except FileNotFoundError:
            # Cleanup worker removed it between exists() and open()
            pass

    # Cache full -> just use original
    cache_size = sum(
        os.path.getsize(
            os.path.join(cache_dir, f)
        )
        for f in os.listdir(cache_dir)
    )

    if cache_size >= max_cache_mb * 1024**2:
        return rasterio.open(filename)

    # Decode and create cached GeoTIFF
    with rasterio.open(filename) as src:
        data = src.read(1)

        write_atomic(cached,data)

    return rasterio.open(cached)

def osm_tile_bounds(x, y, z):
    """
    Returns:
        west, south, east, north
    """

    n = 2 ** z

    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    north = math.degrees(
        math.atan(
            math.sinh(
                math.pi * (1 - 2*y/n)
            )
        )
    )

    south = math.degrees(
        math.atan(
            math.sinh(
                math.pi * (1 - 2*(y+1)/n)
            )
        )
    )

    return west, south, east, north

def srtm_where(lon, lat):
    """
    Determine SRTM tile and pixel position.

    Returns:
        filename,
        x,
        y
    """

    lon0 = math.floor(lon)
    lat0 = math.floor(lat)


    filename = (
        ("N" if lat0 >= 0 else "S")
        +
        f"{abs(lat0):02d}"
        +
        ("E" if lon0 >= 0 else "W")
        +
        f"{abs(lon0):03d}"
        +
        ".SRTMGL1.jp2"
    )


    # SRTMGL1 is 1 arc-second
    size = 3601


    x = int(
        (lon - lon0)
        *
        (size-1)
    )


    # upper-left origin
    y = int(
        (lat0 + 1 - lat)
        *
        (size-1)
    )


    return filename, x, y

def determine_srtm_layout(bounds):

    west,south,east,north = bounds

    # SRTM tiles are 1° x 1°
    if (east - west) > 1.0 or (north - south) > 1.0:
        raise ValueError(
            f"Tile too large for SRTM layout: "
            f"{east-west:.3f}° x {north-south:.3f}°. "
            "Use zoom level >= 9."
        )


    corners = [
        ("NW", west, north),
        ("NE", east, north),
        ("SW", west, south),
        ("SE", east, south)
    ]


    locations = {}

    for name,lon,lat in corners:

        filename,x,y = srtm_where(
            lon,
            lat
        )

        locations[name] = {
            "file": filename,
            "x": x,
            "y": y
        }

    files = {
        v["file"]
        for v in locations.values()
    }


    if len(files) == 1:
        case = "single"


    elif (
        locations["NW"]["file"]
        ==
        locations["SW"]["file"]
        and
        locations["NE"]["file"]
        ==
        locations["SE"]["file"]
    ):
        case = "left_right"


    elif (
        locations["NW"]["file"]
        ==
        locations["NE"]["file"]
        and
        locations["SW"]["file"]
        ==
        locations["SE"]["file"]
    ):
        case = "up_down"


    elif len(files) == 4:
        case = "four"


    else:
        raise RuntimeError(
            "Unexpected SRTM layout"
        )

    return case, locations

def read_osm_dem_tile(bounds, srtm_tiles):

    case, corners = determine_srtm_layout(bounds)

    NW = corners["NW"]
    NE = corners["NE"]
    SW = corners["SW"]
    SE = corners["SE"]

    tile_size = 3601


    if case == "single":

        filename = os.path.join(
            srtm_tiles,
            NW["file"]
        )

        with open_cached_raster(filename) as src:

            x0 = min(NW["x"], SW["x"])
            x1 = max(NE["x"], SE["x"])

            y0 = min(NW["y"], NE["y"])
            y1 = max(SW["y"], SE["y"])

            dem = src.read(
                1,
                window=Window(
                    x0,
                    y0,
                    x1-x0,
                    y1-y0
                )
            )


    elif case == "left_right":

        # western SRTM tile
        west_file = os.path.join(
            srtm_tiles,
            NW["file"]
        )

        east_file = os.path.join(
            srtm_tiles,
            NE["file"]
        )


        # vertical range is identical
        y0 = min(NW["y"], NE["y"])
        y1 = max(SW["y"], SE["y"])


        # left part
        with open_cached_raster(west_file) as src:

            left = src.read(
                1,
                window=Window(
                    NW["x"],
                    y0,
                    tile_size - NW["x"],
                    y1-y0
                )
            )


        # right part
        with open_cached_raster(east_file) as src:

            right = src.read(
                1,
                window=Window(
                    0,
                    y0,
                    NE["x"],
                    y1-y0
                )
            )


        dem = np.hstack(
            (
                left,
                right
            )
        )

    elif case == "up_down":

        north_file = os.path.join(
            srtm_tiles,
            NW["file"]
        )

        south_file = os.path.join(
            srtm_tiles,
            SW["file"]
        )


        x0 = min(
            NW["x"],
            SW["x"]
        )

        x1 = max(
            NE["x"],
            SE["x"]
        )


        # north part
        with open_cached_raster(north_file) as src:

            north = src.read(
                1,
                window=Window(
                    x0,
                    NW["y"],
                    x1-x0,
                    tile_size-NW["y"]
                )
            )


        # south part
        with open_cached_raster(south_file) as src:

            south = src.read(
                1,
                window=Window(
                    x0,
                    0,
                    x1-x0,
                    SW["y"]
                )
            )


        dem = np.vstack(
            (
                north,
                south
            )
        )


    elif case == "four":

        # order:
        #
        # NW | NE
        # ---+---
        # SW | SE


        nw_file = os.path.join(
            srtm_tiles,
            NW["file"]
        )

        ne_file = os.path.join(
            srtm_tiles,
            NE["file"]
        )

        sw_file = os.path.join(
            srtm_tiles,
            SW["file"]
        )

        se_file = os.path.join(
            srtm_tiles,
            SE["file"]
        )


        # NW crop
        with open_cached_raster(nw_file) as src:

            nw = src.read(
                1,
                window=Window(
                    NW["x"],
                    NW["y"],
                    tile_size-NW["x"],
                    tile_size-NW["y"]
                )
            )


        # NE crop
        with open_cached_raster(ne_file) as src:

            ne = src.read(
                1,
                window=Window(
                    0,
                    NE["y"],
                    NE["x"],
                    tile_size-NE["y"]
                )
            )


        # SW crop
        with open_cached_raster(sw_file) as src:

            sw = src.read(
                1,
                window=Window(
                    SW["x"],
                    0,
                    tile_size-SW["x"],
                    SW["y"]
                )
            )


        # SE crop
        with open_cached_raster(se_file) as src:

            se = src.read(
                1,
                window=Window(
                    0,
                    0,
                    SE["x"],
                    SE["y"]
                )
            )


        top = np.hstack(
            (
                nw,
                ne
            )
        )

        bottom = np.hstack(
            (
                sw,
                se
            )
        )


        dem = np.vstack(
            (
                top,
                bottom
            )
        )

    else:
        raise RuntimeError(
            f"Unknown layout {case}"
        )

    dem = dem.astype(np.float32)
    dem[dem >= 65000] = np.nan

    return dem