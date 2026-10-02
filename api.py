import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
)

from dotenv import load_dotenv
from send_apistack_mail import EmailAPIError, send_apistack_mail

from jwt_helpers import (
    encodeJWT,
    decodeJWT,
    verifyJWT,
    set_http_only_cookie,
)

from api_keys import create_api_key
from terrain_api import router as terrain_router


app = FastAPI()

app.include_router(terrain_router)

load_dotenv()

APISTACK_EMAIL_API_KEY = os.environ["APISTACK_EMAIL_API_KEY"]


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
            origin="https://fieldslope.apistack.eu",
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
# Static files
# -------------------------

@app.get(
    "/{full_path:path}",
    tags=["static_files_frontend"]
)
async def catch_all(
    request: Request,
    full_path: str
):
    directory = os.path.abspath("static")
    requested_path = os.path.abspath(
        os.path.join(directory, full_path)
    )

    if full_path.startswith("api/"):
        raise HTTPException(
            status_code=404,
            detail="API endpoint not found"
        )

    if not requested_path.startswith(directory):
        raise HTTPException(
            status_code=403,
            detail="Access forbidden"
        )

    if os.path.isfile(requested_path):
        return FileResponse(requested_path)

    if "." in os.path.basename(full_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        os.path.join(directory, "index.html")
    )
