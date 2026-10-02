import time

import httpx

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

class EmailAPIError(Exception):
    def __init__(self, code, message, status_code=502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

async def send_apistack_mail(
    subject,
    body,
    recipient,
    api_key,
    captcha_token,
):
    """
    Send an email through the APIStack email service.

    Returns:
        dict: JSON response from the email service on success.

    Raises:
        EmailAPIError:
            For any expected email-service failure. Contains:
            - code
            - message
            - status_code
    """

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://email.apistack.eu/api/send_mail",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://captcha.apistack.eu",
                },
                json={
                    "recipient": recipient,
                    "subject": subject,
                    "body": body,
                    "api_key": api_key,
                    "captcha_token": captcha_token,
                },
            )

    except httpx.RequestError as exc:
        raise EmailAPIError(
            code="MAIL_SERVICE_UNAVAILABLE",
            message="Email service is unavailable.",
            status_code=502,
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise EmailAPIError(
            code="MAIL_INVALID_RESPONSE",
            message="Email service did not return valid JSON. Returned: " + str(response),
            status_code=502,
        ) from exc

    if not 200 <= response.status_code < 300:
        detail = data.get("detail", {})

        if isinstance(detail, dict):
            code = detail.get(
                "code",
                "MAIL_SERVICE_ERROR",
            )
            message = detail.get(
                "message",
                "Email service returned an error.",
            )
        else:
            code = "MAIL_SERVICE_ERROR"
            message = str(detail)

        raise EmailAPIError(
            code=code,
            message=message,
            status_code=response.status_code,
        )

    return data