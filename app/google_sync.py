"""
Sync Gmail, Google Calendar, and Google Photos metadata into a local text file for RAG.

Requires a Google Cloud OAuth 2.0 "Desktop app" client JSON saved as credentials.json
(see project README or comments below). First run opens a browser for consent.

APIs to enable in Google Cloud Console: Gmail API, Google Calendar API, Photos Library API.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Read-only access to index text locally; answers still run offline via your SLM.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly",
]


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_credentials_path() -> str:
    return os.environ.get(
        "GOOGLE_OAUTH_CREDENTIALS",
        os.path.join(_project_root(), "credentials.json"),
    )


def default_token_path() -> str:
    return os.environ.get(
        "GOOGLE_OAUTH_TOKEN",
        os.path.join(_project_root(), "data", "google_token.json"),
    )


def default_output_path() -> str:
    return os.environ.get(
        "GOOGLE_SYNC_OUTPUT",
        os.path.join(_project_root(), "data", "google_personal_sync.txt"),
    )


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_credentials() -> Credentials:
    cred_path = default_credentials_path()
    token_path = default_token_path()
    creds: Optional[Credentials] = None
    if os.path.isfile(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.isfile(cred_path):
                raise FileNotFoundError(
                    f"Missing OAuth client file: {cred_path}\n"
                    "Create OAuth 2.0 Desktop credentials in Google Cloud Console, "
                    "download JSON, and save as this path (or set GOOGLE_OAUTH_CREDENTIALS)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
        _ensure_parent(token_path)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds


def _fetch_gmail_lines(service: Any, max_messages: int = 35) -> list[str]:
    lines: list[str] = []
    try:
        lst = (
            service.users()
            .messages()
            .list(userId="me", maxResults=max_messages)
            .execute()
        )
        for msg in lst.get("messages", [])[:max_messages]:
            m = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"],
                )
                .execute()
            )
            hdr = {}
            for h in m.get("payload", {}).get("headers", []) or []:
                name = (h.get("name") or "").lower()
                if name in ("subject", "from", "date"):
                    hdr[name] = (h.get("value") or "")[:300]
            subj = hdr.get("subject", "")
            frm = hdr.get("from", "")
            snippet = (m.get("snippet") or "").replace("\n", " ")[:400]
            if subj or snippet:
                who = f" from {frm}" if frm else ""
                lines.append(f"Email: {subj}{who} — {snippet}".strip(" —"))
    except HttpError as e:
        lines.append(f"(Gmail sync skipped: {e.status_code} {e.reason})")
    return lines


def _fetch_calendar_lines(service: Any, days_back: int = 30, days_forward: int = 90) -> list[str]:
    lines: list[str] = []
    try:
        now = datetime.now(timezone.utc)
        tmin = (now - timedelta(days=days_back)).isoformat()
        tmax = (now + timedelta(days=days_forward)).isoformat()
        ev = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=tmin,
                timeMax=tmax,
                maxResults=80,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        for e in ev.get("items", []):
            summ = e.get("summary", "(no title)")
            loc = e.get("location") or ""
            start = e.get("start", {})
            end = e.get("end", {})
            s = start.get("dateTime") or start.get("date", "")
            en = end.get("dateTime") or end.get("date", "")
            extra = f" @ {loc}" if loc else ""
            lines.append(f"Calendar: {summ}{extra} — {s} → {en}")
    except HttpError as err:
        lines.append(f"(Calendar sync skipped: {err.status_code} {err.reason})")
    return lines


def _fetch_photos_lines(service: Any, page_size: int = 40) -> list[str]:
    lines: list[str] = []
    try:
        out = service.mediaItems().list(pageSize=page_size).execute()
        for it in out.get("mediaItems", []):
            fn = it.get("filename", "photo")
            meta = it.get("mediaMetadata", {}) or {}
            when = meta.get("creationTime", "")
            w = meta.get("width")
            h = meta.get("height")
            dim = f"{w}x{h}" if w and h else ""
            lines.append(f"Photo: {fn} — taken {when} {dim}".strip())
    except HttpError as err:
        lines.append(f"(Photos sync skipped: {err.status_code} {err.reason})")
    return lines


def sync_google_data(
    output_path: Optional[str] = None,
    max_gmail: int = 35,
) -> str:
    """
    Fetch Gmail snippets, Calendar events, and Photos metadata; write one UTF-8 text file.
    Returns path written.
    """
    out = output_path or default_output_path()
    creds = get_credentials()

    chunks: list[str] = [
        "# Synced from Google (Gmail, Calendar, Photos). Do not commit if sensitive.",
        f"# Generated at {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Gmail (recent)",
    ]

    gmail = build("gmail", "v1", credentials=creds)
    chunks.extend(_fetch_gmail_lines(gmail, max_messages=max_gmail))

    chunks.append("")
    chunks.append("## Google Calendar")
    cal = build("calendar", "v3", credentials=creds)
    chunks.extend(_fetch_calendar_lines(cal))

    chunks.append("")
    chunks.append("## Google Photos (metadata)")
    try:
        photos = build(
            "photoslibrary",
            "v1",
            credentials=creds,
            static_discovery=False,
        )
        chunks.extend(_fetch_photos_lines(photos))
    except Exception as e:
        chunks.append(f"(Photos Library API unavailable: {e})")

    text = "\n".join(chunks) + "\n"
    _ensure_parent(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


def main() -> None:
    try:
        path = sync_google_data()
        print(f"Wrote synced personal data to: {path}")
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Sync failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
