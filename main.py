from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from run_pipeline import detect_anomaly_from_raw

app = FastAPI(title="Log Anomaly Detection API")

class DetectRequest(BaseModel):
    raw_log_data: str
    seq_threshold: float = 0.2

@app.post("/detect")
def detect(req: DetectRequest):
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

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000

