import numpy as np
from tqdm import tqdm
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib

def hillshade_advanced(
    dem,
    sigma=1.0,
    z_factor=1.0,
    multidirectional=True,
    directions = (315, 315, 315, 45, 135, 225),
    pad = 10
):
    """
    Cartographic hillshade similar to ArcGIS multidirectional hillshade.
    Returns float image in range [0, 1].
    """
    dem = np.pad(
        dem,
        pad_width = pad,
        mode="edge"
    )

    # Smooth DEM slightly
    dem = gaussian_filter(dem, sigma=sigma)

    # Horn-style gradients (3x3 weighted derivative)
    dzdx = (
        np.roll(dem, -1, axis=1)
        - np.roll(dem, 1, axis=1)
    ) / 2.0

    dzdy = (
        np.roll(dem, -1, axis=0)
        - np.roll(dem, 1, axis=0)
    ) / 2.0

    dzdx *= z_factor
    dzdy *= z_factor

    # slope and aspect
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))

    aspect = np.arctan2(
        -dzdx,
        dzdy
    )

    def illumination(azimuth, altitude=45):
        az = np.deg2rad(azimuth)
        alt = np.deg2rad(altitude)

        shade = (
            np.cos(alt) * np.cos(slope)
            +
            np.sin(alt)
            * np.sin(slope)
            * np.cos(az - aspect)
        )

        return np.clip(shade, 0, 1)

    if multidirectional:
        # Six illumination directions
        shades = [
            illumination(angle)
            for angle in directions
        ]

        shade = np.mean(shades, axis=0)

    else:
        shade = illumination(315)

    # contrast stretch
    shade = np.clip(shade, 0, 1)

    shade = shade[pad:-pad, pad:-pad]

    return shade

def hillshade(heightmap, sigma=1.0, azimuth=315, altitude=45):
    # Smooth DEM
    dem = gaussian_filter(heightmap, sigma=sigma)

    # Gradient
    dy, dx = np.gradient(dem)

    # Surface normal
    nx = -dx
    ny = -dy
    nz = np.ones_like(dem)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    # Light direction
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)

    lx = np.cos(alt) * np.sin(az)
    ly = np.cos(alt) * np.cos(az)
    lz = np.sin(alt)

    # Lambertian shading
    shade = nx * lx + ny * ly + nz * lz
    shade = np.clip(shade, 0, 1)

    return shade


def render_contours(
        heightmap,
        contour_step=10
):
    img = np.zeros(
        heightmap.shape,
        dtype=np.uint8
    )

    minimum = np.nanmin(heightmap)
    maximum = np.nanmax(heightmap)

    levels = np.arange(
        np.floor(minimum / contour_step) * contour_step,
        maximum,
        contour_step
    )

    for level in levels:

        lines = find_contours(
            heightmap,
            level
        )

        for line in lines:

            pixels = line.astype(np.int32)

            for y, x in pixels:

                if (
                    0 <= y < img.shape[0]
                    and
                    0 <= x < img.shape[1]
                ):
                    img[y, x] = 255

    return img

def hillshade_contours(
    heightmap,
    contour_step=10,
    sigma=2.0,
    azimuth=315,
    altitude=45,
    background=235,
    strength=50,
    method="advanced",
):
    if method == "advanced":
        shade = hillshade_advanced(
            heightmap,
            sigma=sigma,
            multidirectional=True,
        )
    else:
        shade = hillshade(
            heightmap,
            sigma=sigma,
            azimuth=azimuth,
            altitude=altitude,
        )

    img = (
        background
        - strength * shade
    ).astype(np.uint8)

    contours = render_contours(
        heightmap,
        contour_step=contour_step,
    )

    mask = contours > 0

    img[mask] = (
        0.7 * img[mask]
        + 0.3 * 80
    ).astype(np.uint8)

    return img

def render_colormap(
    heightmap,
    cmap="terrain",
    vmin=-1,
    vmax=2000,
):

    normalized = (
        heightmap - vmin
    ) / (
        vmax - vmin
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    color_map = matplotlib.colormaps[cmap]

    rgba = color_map(
        normalized
    )

    rgb = (
        rgba[:, :, :3] * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return rgb


def render_heightmap(
        heightmap,
        method="contour",
        cmap="terrain",
        contour_step=10,
        smoothing=0
):
    """
    Render a heightmap.

    Parameters
    ----------
    heightmap : np.ndarray
        Elevation array.

    method : str
        "contour" or "cmap"

    cmap : str
        Matplotlib colormap name.

    contour_step : float
        Contour interval in meters.

    smoothing : float
        Gaussian smoothing sigma in pixels.
    """

    if smoothing > 0:
        heightmap = gaussian_filter(
            heightmap,
            sigma=smoothing
        )

    if method == "contour":
        return hillshade_contours(
            heightmap,
            contour_step=contour_step
        )

    elif method == "cmap":
        return render_colormap(
            heightmap,
            cmap=cmap
        )

    raise ValueError(
        f"Unknown render method: {method}"
    )

def contour_settings_for_zoom(z):
    """
    Returns contour interval (meters) and DEM smoothing sigma (pixels).
    """

    settings = {
        9:(100,  5), 10:(100,  5),
        11:(100,  5), 12:(50,   5), 13:(25,   5),
        14:(20,   5),15:(10,  5),  16:(5,    5),
        17:(2,    5), 18:(1,    5), 19:(1,    5)
    }

    return settings.get(z, (1, 0))