import os
import json
import io
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request

load_dotenv()

router = APIRouter()
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
# default folder id if none provided
DEFAULT_FOLDER_ID = "1LfjlIA_VLou2gfIsvt6nYBoXKN_JutTs"
TOKEN_STORE = Path("tokens.json")


def load_saved_tokens():
    if TOKEN_STORE.exists():
        try:
            return json.loads(TOKEN_STORE.read_text())
        except Exception:
            return {}
    return {}


def save_tokens(tokens: dict):
    try:
        TOKEN_STORE.write_text(json.dumps(tokens))
    except Exception:
        pass


def get_credentials():
    """
    Prefer service account (SERVICE_ACCOUNT_FILE) for non-interactive uploads.
    Fallback to persisted user tokens (tokens.json). Returns credentials or None.
    """
    sa_file = os.getenv("SERVICE_ACCOUNT_FILE")
    if sa_file and Path(sa_file).exists():
        return ServiceAccountCredentials.from_service_account_file(sa_file, scopes=SCOPES)

    saved = load_saved_tokens()
    access = saved.get("access_token")
    refresh = saved.get("refresh_token")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not refresh and not access:
        return None

    creds = Credentials(token=access, refresh_token=refresh, client_id=client_id, client_secret=client_secret, scopes=SCOPES)
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            saved["access_token"] = creds.token
            if creds.refresh_token:
                saved["refresh_token"] = creds.refresh_token
            save_tokens(saved)
    except Exception:
        # If refresh fails, still return what we have (caller will see failure when building service)
        pass

    return creds


@router.post("/upload")
async def upload_file_to_drive(
    file: UploadFile = File(...),
    folder_id: str = Form(DEFAULT_FOLDER_ID),
):
    """
    Upload a file to Google Drive.
    Uses SERVICE_ACCOUNT_FILE if set, otherwise uses persisted user tokens in tokens.json.
    Default folder_id used when not provided.
    """
    creds = get_credentials()
    if creds is None:
        raise HTTPException(status_code=401, detail="No credentials available. Set SERVICE_ACCOUNT_FILE or authenticate and store tokens.")

    try:
        drive_service = build("drive", "v3", credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build Drive service: {e}")

    file_content = await file.read()
    file_stream = io.BytesIO(file_content)

    media = MediaIoBaseUpload(file_stream, mimetype=file.content_type or "application/octet-stream", resumable=False)

    file_metadata = {
        "name": file.filename,
        "parents": [folder_id] if folder_id else []
    }

    try:
        uploaded = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, mimeType, parents"
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return JSONResponse(uploaded)

