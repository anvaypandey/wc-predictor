from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.state import load_state
from backend.routers import predict, simulate, accuracy


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="WC Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,  prefix="/api")
app.include_router(simulate.router, prefix="/api")
app.include_router(accuracy.router, prefix="/api")

# Serve React build in production
_dist = Path("frontend/dist")
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
