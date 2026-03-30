from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.api.database import init_db
from apps.api.routers import upload, screening

init_db()

app = FastAPI(title="Systematic Review API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(screening.router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Systematic Review API is running"}
