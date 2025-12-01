import os
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request

load_dotenv()

router = APIRouter()
SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/documents"]
TOKEN_STORE = Path("tokens.json")


class CreateFileRequest(BaseModel):
    title: str
    content: str
    folder_id: str = None


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
    Prefer service account (SERVICE_ACCOUNT_FILE) for non-interactive creation.
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
        pass

    return creds


@router.post("/create-file")
def create_file(request: CreateFileRequest):
    """
    Create a Google Docs file with the given title and content.
    Uses SERVICE_ACCOUNT_FILE if configured, otherwise persisted user tokens.
    """
    creds = get_credentials()
    if creds is None:
        raise HTTPException(status_code=401, detail="No credentials available. Set SERVICE_ACCOUNT_FILE or authenticate and store tokens.")

    try:
        drive_service = build("drive", "v3", credentials=creds)
        docs_service = build("docs", "v1", credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build Google services: {e}")

    file_metadata = {
        "name": request.title,
        "mimeType": "application/vnd.google-apps.document"
    }
    if request.folder_id:
        file_metadata["parents"] = [request.folder_id]

    try:
        created_file = drive_service.files().create(
            body=file_metadata,
            fields="id, name, mimeType, parents"
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create file: {e}")

    # Insert content into the created Google Doc
    try:
        requests_body = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": request.content
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=created_file["id"],
            body={"requests": requests_body}
        ).execute()
    except Exception as e:
        # If docs update fails, return created file info but note the error
        return JSONResponse({"created_file": created_file, "docs_update_error": str(e)}, status_code=207)

    return JSONResponse(created_file)

