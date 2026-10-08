from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ai_model import analyze_project_idea

app = FastAPI(title="Project Management API", version="0.1.0")
CRACK_MODEL_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "crack_classifier.pth"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    status: Literal["Planned", "In Progress", "Completed"] = "Planned"
    priority: Literal["Low", "Medium", "High"] = "Medium"


class ProjectUpdate(ProjectCreate):
    pass


class ProjectRecord(ProjectCreate):
    id: int
    created_at: str


class ProjectAnalysisRequest(BaseModel):
    title: str = Field(..., min_length=3)
    domain: str = Field(default="AI/ML")
    description: str = Field(..., min_length=10)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running"}


app.state.projects = [
    {
        "id": 1,
        "title": "Project Repository Setup",
        "description": "Initialize the repository, install tooling, and create a working backend and frontend scaffold.",
        "status": "Completed",
        "priority": "High",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": 2,
        "title": "AI Workflow Dashboard",
        "description": "Build a dashboard to review project feasibility, task progress, and implementation priorities.",
        "status": "In Progress",
        "priority": "High",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]


def get_project_stats():
    projects = app.state.projects
    total = len(projects)
    planned = sum(1 for p in projects if p["status"] == "Planned")
    in_progress = sum(1 for p in projects if p["status"] == "In Progress")
    completed = sum(1 for p in projects if p["status"] == "Completed")
    high_priority = sum(1 for p in projects if p["priority"] == "High")

    return {
        "total_projects": total,
        "planned": planned,
        "in_progress": in_progress,
        "completed": completed,
        "high_priority": high_priority,
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Project management API is running"}


@app.post("/api/analyze")
async def analyze_project(project: ProjectAnalysisRequest):
    return analyze_project_idea(
        title=project.title,
        domain=project.domain,
        description=project.description,
    )


@app.post("/api/crack/predict")
async def predict_concrete_crack(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a supported image file")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        from app.crack_model import predict_crack_image

        result = predict_crack_image(image_bytes, CRACK_MODEL_CHECKPOINT)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Crack model is not trained yet. Run backend/train_crack_model.py first.",
        ) from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image") from error

    return {"filename": image.filename, **result}


@app.get("/api/projects")
async def list_projects():
    return app.state.projects


@app.get("/api/stats")
async def get_stats():
    return get_project_stats()


@app.post("/api/projects", response_model=ProjectRecord)
async def create_project(project: ProjectCreate):
    new_id = max((p["id"] for p in app.state.projects), default=0) + 1
    new_item = {
        "id": new_id,
        "title": project.title,
        "description": project.description,
        "status": project.status,
        "priority": project.priority,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    app.state.projects.insert(0, new_item)
    return new_item


@app.get("/api/projects/{project_id}", response_model=ProjectRecord)
async def get_project(project_id: int):
    project = next((p for p in app.state.projects if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.put("/api/projects/{project_id}", response_model=ProjectRecord)
async def update_project(project_id: int, project: ProjectUpdate):
    for index, current in enumerate(app.state.projects):
        if current["id"] == project_id:
            updated = {
                **current,
                "title": project.title,
                "description": project.description,
                "status": project.status,
                "priority": project.priority,
            }
            app.state.projects[index] = updated
            return updated
    raise HTTPException(status_code=404, detail="Project not found")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: int):
    initial_count = len(app.state.projects)
    app.state.projects = [p for p in app.state.projects if p["id"] != project_id]
    if len(app.state.projects) == initial_count:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "id": project_id}
