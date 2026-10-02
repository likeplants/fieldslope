import os
import time
from typing import Dict

import jwt
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = os.environ["JWT_ALGORITHM"]


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

    token = jwt_helpers.encode(
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


def set_http_only_cookie(response, key, value, debug=False):
    if not debug:
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
