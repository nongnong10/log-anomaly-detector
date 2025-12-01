import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import RedirectResponse, JSONResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import io

from pydantic import BaseModel

load_dotenv()

app = FastAPI()
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

CLIENT_SECRETS_FILE = "oauth_client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Lưu token tạm thời (production dùng DB)
tokens = {}

# ===== OAuth Login =====
@app.get("/auth/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return RedirectResponse(auth_url)

# ===== OAuth Callback =====
@app.get("/auth/callback")
def auth_callback(code: str):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials

    tokens["access_token"] = credentials.token
    tokens["refresh_token"] = credentials.refresh_token

    return JSONResponse({"message": "Login successful", "access_token": credentials.token})

# ===== Upload File =====
@app.post("/upload")
async def upload_file_to_drive(
        file: UploadFile = File(...),
        folder_id: str = Form(...)
):
    if "access_token" not in tokens:
        return JSONResponse({"error": "User not logged in"}, status_code=401)

    # Tạo credentials từ access token
    creds = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES
    )

    drive_service = build("drive", "v3", credentials=creds)

    # Đọc file
    file_content = await file.read()
    file_stream = io.BytesIO(file_content)

    media = MediaIoBaseUpload(
        file_stream,
        mimetype=file.content_type,
        resumable=False
    )

    # Metadata file
    file_metadata = {
        "name": file.filename,
        "parents": [folder_id]  # Upload vào folder cụ thể
    }

    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, mimeType, parents"
    ).execute()

    return uploaded

# Pydantic model cho input JSON
class CreateFileRequest(BaseModel):
    title: str
    content: str
    folder_id: str = None  # optional, nếu muốn tạo trong folder cụ thể

@app.post("/create-file")
def create_file(request: CreateFileRequest):
    if "access_token" not in tokens:
        return JSONResponse({"error": "User not logged in"}, status_code=401)

    # Tạo credentials từ access token
    creds = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES
    )

    drive_service = build("drive", "v3", credentials=creds)

    # Metadata để tạo file Google Docs
    file_metadata = {
        "name": request.title,
        "mimeType": "application/vnd.google-apps.document"
    }

    if request.folder_id:
        file_metadata["parents"] = [request.folder_id]

    # Tạo file Google Docs trống
    created_file = drive_service.files().create(
        body=file_metadata,
        fields="id, name, mimeType, parents"
    ).execute()

    # Update nội dung file
    docs_service = build("docs", "v1", credentials=creds)
    requests = [
        {
            "insertText": {
                "location": {"index": 1},
                "text": request.content
            }
        }
    ]

    docs_service.documents().batchUpdate(
        documentId=created_file["id"],
        body={"requests": requests}
    ).execute()

    return created_file