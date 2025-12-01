from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from google_auth_oauthlib.flow import Flow
from pathlib import Path
from dotenv import load_dotenv
import os
import json

load_dotenv()

router = APIRouter(prefix="/auth")
CLIENT_SECRETS_FILE = os.getenv("CLIENT_SECRETS_FILE", "./oauth_client_secret.json")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
REDIRECT_URI = os.getenv("REDIRECT_URI")
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


@router.get("/login")
def login():
    """
    Redirect user to Google OAuth consent screen.
    In production add a state param to protect against CSRF.
    """
    if not Path(CLIENT_SECRETS_FILE).exists():
        raise HTTPException(status_code=500, detail=f"Missing client secrets file: {CLIENT_SECRETS_FILE}")
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return RedirectResponse(auth_url)


@router.get("/callback")
def auth_callback(code: str = None):
    """
    Exchange code for tokens and persist them to tokens.json.
    """
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials
    tokens = load_saved_tokens()
    tokens["access_token"] = credentials.token
    # refresh_token may be None on subsequent approvals; keep existing if present
    if credentials.refresh_token:
        tokens["refresh_token"] = credentials.refresh_token
    save_tokens(tokens)
    return JSONResponse({"message": "Login successful"})

