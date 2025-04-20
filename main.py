from fastapi import FastAPI
from app.api import router
import uvicorn
import yaml

app = FastAPI()

# Include API routes
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Court-Judgment Q&A System is running."}

if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    
    uvicorn.run(
        "app.api:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )