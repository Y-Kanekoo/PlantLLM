import os

import uvicorn

from app_simple import app

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )
