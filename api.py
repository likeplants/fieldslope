import io
import json
import os
import random
import secrets
import smtplib
import tempfile
import threading
import time

from email.mime.text import MIMEText
from typing import Dict
import send_apistack_mail
from send_apistack_mail import EmailAPIError

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from smtplib import (
    SMTPException,
    SMTPAuthenticationError,
    SMTPConnectError,
    SMTPSenderRefused,
)

import jwt
from dotenv import load_dotenv

from get_tiles import osm_tile_bounds, read_osm_dem_tile
from rendering import hillshade_contours, contour_settings_for_zoom, render_colormap

app = FastAPI()

load_dotenv()

JWT_SECRET = os.environ["secret"]
JWT_ALGORITHM = os.environ["algorithm"]
APISTACK_EMAIL_API_KEY = os.environ["APISTACK_EMAIL_API_KEY"]

DEBUG = False
API_KEYS_FILE = "api_keys.json"

def token_response(token: str):
    return {
        "access_token": token
    }


def encodeJWT(payload: dict) -> Dict[str, str]:
    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )


def signJWT(payload: dict) -> Dict[str, str]:
    payload["expires"] = time.time() + 6

    if "password" in payload:
        del payload["password"]

    token = jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )

    return token_response(token)


def decodeJWT(token: str, ignore_expiration: bool) -> dict:
    try:
        decoded_token = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": False}
        )

        if ignore_expiration:
            return decoded_token

        return (
            decoded_token
            if decoded_token.get("expires", 0) >= time.time()
            else None
        )

    except Exception:
        return {}


def verifyJWT(
    jwtoken: str,
    condition: dict,
    ignore_expiration: bool = False
) -> bool:

    try:
        payload = decodeJWT(
            jwtoken,
            ignore_expiration
        )

        if not payload:
            return False

        for k, v in condition.items():
            if payload.get(k) != v:
                return False

        return True

    except Exception:
        return False


def set_http_only_cookie(response, key, value):
    if not DEBUG:
        response.set_cookie(
            key=key,
            value=value,
            httponly=True,
            secure=True,
            samesite="strict"
        )
    else:
        response.set_cookie(
            key=key,
            value=value,
            httponly=True
        )

    return response


def send_gmail(subject, body, sender, recipients, password):

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp_server:

            smtp_server.login(
                sender,
                password
            )

            smtp_server.sendmail(
                sender,
                recipients,
                msg.as_string()
            )

        return True

    except (
        SMTPAuthenticationError,
        SMTPConnectError,
        SMTPSenderRefused,
        SMTPException,
    ) as e:

        print(e)
        return False

    except Exception as e:
        print(e)
        return False


# -------------------------
# Simple JSON database
# -------------------------

api_keys_lock = threading.Lock()


def load_api_keys():

    try:
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        return {}


def save_api_keys(data, retries=5):

    for attempt in range(retries):

        try:
            directory = os.path.dirname(API_KEYS_FILE) or "."

            fd, temp_path = tempfile.mkstemp(
                dir=directory
            )

            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(
                        data,
                        f,
                        indent=4
                    )

                    f.flush()
                    os.fsync(f.fileno())

                os.replace(
                    temp_path,
                    API_KEYS_FILE
                )

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            return True

        except (OSError, IOError):

            if attempt == retries - 1:
                raise

            time.sleep(
                0.2 * (2 ** attempt)
                + random.uniform(0, 0.2)
            )


def create_api_key(email):

    with api_keys_lock:

        api_keys = load_api_keys()

        for key, entry in api_keys.items():

            if entry["email"] == email:
                return key

        api_key = (
            "api_"
            + secrets.token_hex(32)
        )

        api_keys[api_key] = {
            "email": email,
            "created": int(time.time())
        }

        save_api_keys(api_keys)

        return api_key


# -------------------------
# Request API key
# -------------------------

@app.post(
    "/api/request_api_key",
    tags=["api_management"]
)
async def request_api_key(request: Request):

    data = await request.json()

    email = data.get("email")
    captcha_token = data.get("captcha_token")

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email must be provided"
        )

    if not captcha_token:
        raise HTTPException(
            status_code=400,
            detail="CAPTCHA must be completed"
        )

    access_token = encodeJWT({
        "email": email,
        "purpose": "create_api_key",
        "expires": time.time() + 3600
    })

    url = (
        f"https://{request.url.netloc}"
        "/api/api_key_redirect?access_token="
        + access_token
    )

    try:
        await send_apistack_mail(
            subject="Your fieldslope API key request",
            body=(
                "Click this link to create your API key:\n\n"
                + url
                + "\n\nThis link expires in one hour."
            ),
            recipient=email,
            api_key=APISTACK_EMAIL_API_KEY,
            captcha_token=captcha_token,
        )

    except EmailAPIError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
            },
        )

    return JSONResponse({
        "detail": (
            "If the email exists, a verification link was sent."
        )
    })


# -------------------------
# Redirect
# -------------------------

@app.get(
    "/api/api_key_redirect",
    tags=["api_management"]
)
def api_key_redirect(access_token: str = ""):

    response = RedirectResponse(
        "/api_key.html",
        status_code=303
    )

    return set_http_only_cookie(
        response,
        "access_token",
        access_token
    )


# -------------------------
# Create/display API key
# -------------------------

@app.get(
    "/api/get_api_key",
    tags=["api_management"]
)
def get_api_key(request: Request):

    access_token = request.cookies.get(
        "access_token"
    )

    if not access_token:
        raise HTTPException(401)

    if not verifyJWT(access_token, {}):
        raise HTTPException(401)

    payload = decodeJWT(
        access_token,
        True
    )

    if payload.get("purpose") != "create_api_key":
        raise HTTPException(401)

    api_key = create_api_key(
        payload["email"]
    )

    response = JSONResponse({
        "api_key": api_key
    })

    response.delete_cookie(
        "access_token"
    )

    return response


# -------------------------
# API key authentication
# -------------------------

def verify_api_key(
    authorization: str = Header(None)
):

    if not authorization:
        raise HTTPException(401)

    if not authorization.startswith("Bearer "):
        raise HTTPException(401)

    api_key = authorization[7:].strip()

    entry = load_api_keys().get(
        api_key
    )

    if not entry:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return {
        "email": entry["email"]
    }


# -------------------------
# Example protected API
# -------------------------

from fastapi import Depends
from fastapi.responses import Response
from PIL import Image
import io
import numpy as np
from scipy.ndimage import zoom


def verify_api_key(
    token: str = Query(None)
):

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    entry = load_api_keys().get(
        token
    )

    if not entry:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return {
        "email": entry["email"]
    }


# -------------------------
# Protected tile endpoint
# -------------------------

from fastapi.responses import Response
from PIL import Image
import io
from scipy.ndimage import zoom


@app.get(
    "/api/tiles/shaded_contour/tile",
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

@app.get(
    "/api/terrain/window",
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
    # Approximate maximum: one zoom 9 tile
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

        raw = np.asarray(
            dem
        )

        raw = np.clip(
            raw,
            0,
            65535
        ).astype(
            np.uint16
        )


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

        contour_step, smoothing = contour_settings_for_zoom(
            9
        )


        rendered = hillshade_contours(
            dem,
            contour_step=contour_step,
            sigma=smoothing,
        )


        image = Image.fromarray(
            rendered
        )


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
    # Colormap placeholder
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

@app.get(
    "/api/point_slope",
    tags=["terrain"]
)
def point_slope(
    longitude: float,
    latitude: float,
    size: int = 16,
    radius_m: float =30.0,
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
            np.radians(
                latitude
            )
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

    gy, gx = np.gradient(
        dem
    )

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

    # Flat terrain threshold (degrees)
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
            "east_north_east",
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

######################## STATIC FILES #########################
@app.get("/{full_path:path}", tags=["static_files_frontend"])
async def catch_all(request: Request, full_path: str):
    directory = os.path.abspath("static")
    requested_path = os.path.abspath(os.path.join(directory, full_path))

    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    # Ensure the requested file is within the directory to prevent path traversal
    if not requested_path.startswith(directory):
        raise HTTPException(status_code=403, detail="Access forbidden")

    # If the requested path is a file, serve it
    if os.path.isfile(requested_path):
        return FileResponse(requested_path)

    # If it looks like a file but does not exist, return 404
    if "." in os.path.basename(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Otherwise, treat it as an SPA route
    return FileResponse(os.path.join(directory, "index.html"))