from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from run_pipeline import detect_anomaly_from_raw
import os
import psycopg2
from dotenv import load_dotenv
from api.log_lines import router as log_lines_router

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

app = FastAPI(title="Log Anomaly Detection API")
app.include_router(log_lines_router)

@app.on_event("startup")
def startup_db():
    try:
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            sslmode="require",
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        print("DB connection successful.")
        app.state.db_conn = conn
        app.state.db_cursor = cursor
    except Exception as e:
        print(f"DB connection failed: {e}")
        app.state.db_conn = None
        app.state.db_cursor = None

@app.on_event("shutdown")
def shutdown_db():
    try:
        cursor = getattr(app.state, "db_cursor", None)
        conn = getattr(app.state, "db_conn", None)
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("DB connection closed.")
    except Exception as e:
        print(f"Error closing DB connection: {e}")

class DetectRequest(BaseModel):
    raw_log_data: str
    seq_threshold: float = 0.2

@app.post("/detect")
def detect(req: DetectRequest):
    # If you need the cursor: cursor = getattr(app.state, "db_cursor", None)
    try:
        summary = detect_anomaly_from_raw(req.raw_log_data, seq_threshold=req.seq_threshold)
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/db/health")
def db_health():
    if getattr(app.state, "db_conn", None):
        return {"db_status": "connected"}
    return {"db_status": "not_connected"}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
