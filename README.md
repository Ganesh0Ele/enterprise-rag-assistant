# Enterprise RAG Assistant

A local, full-stack FastAPI application for secure company-document search and cited Q&A.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. The first account is automatically an **admin**. Upload a PDF, then ask questions in the chat panel.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The Compose stack uses PostgreSQL and Redis. Local development defaults to SQLite if `DATABASE_URL` is not set.

## Security model

- Passwords are bcrypt-hashed; access tokens are signed JWTs.
- Documents are scoped to a role (`employee`, `manager`, or `admin`). A user may only retrieve chunks permitted for their role.
- PDF upload is restricted to admins and managers. Admins can manage users and document access.
- Answers are generated only from retrieved chunks and include source citations.

Set a strong `SECRET_KEY` and use HTTPS before deploying beyond local development.
