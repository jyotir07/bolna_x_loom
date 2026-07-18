"""FastAPI application entrypoint.

Run with: `uvicorn backend.app:app --reload`
"""

from fastapi import FastAPI

from backend.routes.chat import router as chat_router
from backend.services.logger import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Bolna × Loom Backend", version="0.1.0")
    app.include_router(chat_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
