from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="zxell API Server")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/get-task", response_model=schemas.TrainingDataOut)
def get_task(client_id: str, db: Session = Depends(get_db)):
    task = db.query(models.TrainingData).filter(models.TrainingData.status == "pending").first()
    if not task:
        raise HTTPException(status_code=404, detail="No pending tasks")

    task.status = "processing"
    task.assigned_to = client_id
    db.commit()
    db.refresh(task)
    return task

@app.post("/api/post-result")
def post_result(result: schemas.ResultIn, db: Session = Depends(get_db)):
    task = db.query(models.TrainingData).filter(models.TrainingData.id == result.data_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Data ID not found")
    
    task.status = "done"
    db.commit()
    return {"message": "Result received for zxell", "data_id": result.data_id}

