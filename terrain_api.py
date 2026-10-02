import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api_keys import verify_api_key
from get_tiles import osm_tile_bounds, read_osm_dem_tile
from rendering import (
    contour_settings_for_zoom,
    hillshade_contours,
    render_colormap,
)


router = APIRouter(prefix="/api")


# -------------------------
# Protected tile endpoint
# -------------------------

@router.get(
    "/tiles/shaded_contour/tile",
    tags=["tiles"]
)
def shaded_contour_tile(
    x: int,
    y: int,
    z: int,
    user=Depends(verify_api_key)
):

    bounds = osm_tile_bounds(
        x,
        y,
        z
    )

    try:
        dem = read_osm_dem_tile(
            bounds,
            srtm_tiles="./srtm_tiles"
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

        raise HTTPException(
            status_code=404,
            detail="Tile data not available"
        )

    contour_step, smoothing = contour_settings_for_zoom(z)

    rendered = hillshade_contours(
        dem,
        contour_step=contour_step,
        sigma=smoothing,
    )

    image = Image.fromarray(rendered)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg"
    )


# -------------------------
# Terrain window
# -------------------------

@router.get(
    "/terrain/window",
    tags=["terrain"]
)
def terrain_window(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    format: str = "raw",
    user=Depends(verify_api_key)
):

    # -------------------------
    # Validate requested window
    # -------------------------

    if (
        max_lon - min_lon > 0.7 or
        max_lat - min_lat > 0.7
    ):
        raise HTTPException(
            status_code=400,
            detail="Requested window too large. Maximum size is .7 degrees."
        )

    bounds = (
        min_lon,
        min_lat,
        max_lon,
        max_lat
    )

    try:
        dem = read_osm_dem_tile(
            bounds,
            srtm_tiles="./srtm_tiles"
        )

    except Exception as e:
        print(e)

        raise HTTPException(
            status_code=404,
            detail="Terrain data not available"
        )

    # -------------------------
    # Limit output resolution
    # -------------------------

    max_dimension = 2048

    height, width = dem.shape

    scale = min(
        max_dimension / width,
        max_dimension / height,
        1.0
    )

    if scale < 1.0:
        dem = zoom(
            dem,
            scale,
            order=1
        )

    # -------------------------
    # Raw elevation PNG
    # -------------------------

    if format == "raw":

        raw = np.asarray(dem)

        raw = np.clip(
            raw,
            0,
            65535
        ).astype(np.uint16)

        image = Image.fromarray(
            raw,
            mode="I;16"
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/png"
        )

    # -------------------------
    # Shaded contour JPEG
    # -------------------------

    elif format == "shaded_contours":

        contour_step, smoothing = contour_settings_for_zoom(9)

        rendered = hillshade_contours(
            dem,
            contour_step=contour_step,
            sigma=smoothing,
        )

        image = Image.fromarray(rendered)

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg"
        )

    # -------------------------
    # Colormap
    # -------------------------

    elif format == "colormap":

        arr = render_colormap(
            dem,
            cmap="viridis",
            vmin=np.nanmin(dem),
            vmax=np.nanmax(dem)
        )

        buffer = io.BytesIO()

        image = Image.fromarray(
            arr,
            mode="RGB"
        )

        image.save(
            buffer,
            format="PNG"
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/png"
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use raw, shaded_contours, or colormap."
        )


# -------------------------
# Point slope
# -------------------------

@router.get(
    "/point_slope",
    tags=["terrain"]
)
def point_slope(
    longitude: float,
    latitude: float,
    size: int = 16,
    radius_m: float = 30.0,
    sigma=0,
    user=Depends(verify_api_key)
):

    # -----------------------------------------
    # Determine geographic bounds
    # -----------------------------------------

    dlat = radius_m / 111320.0

    dlon = radius_m / (
        111320.0 *
        np.cos(
            np.radians(latitude)
        )
    )

    bounds = (
        longitude - dlon,
        latitude - dlat,
        longitude + dlon,
        latitude + dlat
    )

    # -----------------------------------------
    # Read DEM
    # -----------------------------------------

    try:
        dem = read_osm_dem_tile(
            bounds,
            srtm_tiles="./srtm_tiles"
        )

    except Exception as e:
        print(e)

        raise HTTPException(
            status_code=404,
            detail="Terrain data not available"
        )

    # -----------------------------------------
    # Resample to fixed size
    # -----------------------------------------

    sy = size / dem.shape[0]
    sx = size / dem.shape[1]

    dem = zoom(
        dem,
        (sy, sx),
        order=1
    )

    # -----------------------------------------
    # Center point
    # -----------------------------------------

    cy = size // 2
    cx = size // 2

    height = float(
        dem[cy, cx]
    )

    # -----------------------------------------
    # Smooth
    # -----------------------------------------

    dem = gaussian_filter(
        dem,
        sigma=sigma
    )

    # -----------------------------------------
    # Compute gradient
    # -----------------------------------------

    gy, gx = np.gradient(dem)

    dzdx = float(
        gx[cy, cx]
    )

    dzdy = float(
        gy[cy, cx]
    )

    # -----------------------------------------
    # Physical pixel spacing
    # -----------------------------------------

    width_m = 2.0 * radius_m

    pixel_size = width_m / size

    dzdx /= pixel_size
    dzdy /= pixel_size

    # -----------------------------------------
    # Slope angle
    # -----------------------------------------

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dzdx * dzdx +
                dzdy * dzdy
            )
        )
    )

    # -----------------------------------------
    # Slope direction
    # -----------------------------------------

    east_gradient = -dzdx
    north_gradient = dzdy

    angle = (
        np.degrees(
            np.arctan2(
                north_gradient,
                east_gradient
            )
        ) + 360.0
    ) % 360.0

    epsilon = 0.1

    if slope < epsilon:
        direction = "no_slope"

    else:

        directions = [
            "east",
            "north_east",
            "north_north_east",
            "north",
            "north_north_west",
            "north_west",
            "west_north_west",
            "west",
            "west_south_west",
            "south_west",
            "south_south_west",
            "south",
            "south_south_east",
            "south_east",
            "east_south_east",
        ]

        index = int(
            (angle + 11.25) // 22.5
        ) % 16

        direction = directions[index]

    # -----------------------------------------
    # Response
    # -----------------------------------------

    return {
        "longitude": longitude,
        "latitude": latitude,
        "height": round(height, 2),
        "slope": round(float(slope), 2),
        "slope_direction": direction
    }