from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import preprocess
from routers import object_segmentation

app = FastAPI(title="3D Model Conversion API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "3d model conversion"}

app.include_router(preprocess.router)
app.include_router(object_segmentation.router)