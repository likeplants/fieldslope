import json
import os
import random
import secrets
import tempfile
import threading
import time

from fastapi import Header, HTTPException


API_KEYS_FILE = "api_keys.json"

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


def verify_api_key(
    token: str = Header(None)
):
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    # Support "Bearer <key>"
    if token.startswith("Bearer "):
        token = token[7:].strip()

    entry = load_api_keys().get(token)

    if not entry:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return {
        "email": entry["email"]
    }