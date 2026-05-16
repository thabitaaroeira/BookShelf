"""
BookShelf - A FastAPI application for managing your book collection.

Features:
- CRUD operations for books
- OCR scanning of book spines
- Review workflow for incomplete OCR results
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.views import router as views_router
from app.database import engine, Base


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    Base.metadata.create_all(bind=engine)
    logger.info("Application ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="BookShelf",
    version="1.0.0",
    description="A book collection manager with OCR scanning capability",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(api_router)
app.include_router(views_router)


@app.get("/")
def root():
    """Redirect root to books list."""
    return RedirectResponse(url="/books", status_code=302)
