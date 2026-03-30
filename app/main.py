from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.views import router as views_router
from app.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="BookShelf", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(api_router)
app.include_router(views_router)


@app.get("/")
def root():
    return RedirectResponse(url="/books", status_code=302)
