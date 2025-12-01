from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.detect_anomaly_sequence import detect_anomaly_from_raw
from api.detect_anomaly_sequence_v2 import detect_anomaly_from_raw_v2  # updated import
import os
import psycopg2
from dotenv import load_dotenv
from api.log_lines import router as log_lines_router
from api.log_sequences import router as log_sequences_router  # new import
from api.create_file import router as create_file_router
from api.upload_file import router as upload_file_router
from api.auth import router as auth_router
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

app = FastAPI(title="Log Anomaly Detection API")

# CORS middleware (adjust origins as needed)
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://192.168.1.5:5173",
    "https://log-anomaly-detectio-dy6d.bolt.host"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(log_lines_router)
app.include_router(log_sequences_router)  # include new sequences router
app.include_router(create_file_router)
app.include_router(upload_file_router)
app.include_router(auth_router)

@app.on_event("startup")
def startup_db():
    """
    Attempt to connect using DB_HOST (e.g. supabase). If that fails,
    attempt a fallback using LOCAL_DB_HOST (default: localhost) with
    LOCAL_DB_USER / LOCAL_DB_PASSWORD / LOCAL_DB_NAME; if those are not
    provided, fall back to defaults:
      host = localhost
      user = postgres
      password = 123456
      dbname = log_anomaly_detector
    """
    primary_host = DB_HOST
    primary_user = DB_USER
    primary_password = DB_PASSWORD
    primary_port = DB_PORT
    primary_dbname = DB_NAME

    # Local fallback settings (can be provided in .env)
    fallback_host = os.getenv("LOCAL_DB_HOST", "localhost")
    fallback_user = os.getenv("LOCAL_DB_USER", "postgres")
    fallback_password = os.getenv("LOCAL_DB_PASSWORD", "123456")
    fallback_port = os.getenv("LOCAL_DB_PORT", primary_port or "5432")
    fallback_dbname = os.getenv("LOCAL_DB_NAME", "log_anomaly_detector")

    conn = None
    cursor = None

    def try_connect(host, user, password, port, dbname):
        try:
            c = psycopg2.connect(
                user=user,
                password=password,
                host=host,
                port=port,
                dbname=dbname,
                sslmode="require" if host != "localhost" and host != "127.0.0.1" else "disable",
            )
            cur = c.cursor()
            cur.execute("SELECT 1;")
            return c, cur
        except Exception as e:
            print(f"DB connect attempt to host={host} user={user} db={dbname} failed: {e}")
            return None, None

    # Try Supabase / primary host first then Try local fallback
    # conn, cursor = try_connect(primary_host, primary_user, primary_password, primary_port, primary_dbname)
    # if conn is None:
    #     print(f"Primary DB host {primary_host} failed, attempting fallback host {fallback_host}")
    #     conn, cursor = try_connect(fallback_host, fallback_user, fallback_password, fallback_port, fallback_dbname)
    conn, cursor = try_connect(fallback_host, fallback_user, fallback_password, fallback_port, fallback_dbname)


    if conn and cursor:
        used_host = cursor.connection.get_dsn_parameters().get("host")
        used_user = cursor.connection.get_dsn_parameters().get("user")
        used_db = cursor.connection.get_dsn_parameters().get("dbname")
        print(f"DB connection successful (host={used_host} user={used_user} db={used_db})")
        app.state.db_conn = conn
        app.state.db_cursor = cursor
    else:
        print("DB connection failed for both primary and fallback hosts.")
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
    notify_slack: bool = False
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

@app.post("/v2/detect")
def detect(req: DetectRequest):
    try:
        conn = getattr(app.state, "db_conn", None)
        summary = detect_anomaly_from_raw_v2(req.raw_log_data, seq_threshold=req.seq_threshold, db_conn=conn, notify_slack=req.notify_slack)
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

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
