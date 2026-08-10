import uvicorn

from dotenv import load_dotenv
import os

load_dotenv(".env")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=True,
        reload=True
    )