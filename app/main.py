from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from passlib.context import CryptContext


ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_assistant.db")
SECRET_KEY = os.getenv("SECRET_KEY", "local-development-secret-change-me")
ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
ROLES = {"employee": 1, "manager": 2, "admin": 3}

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="employee")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    access_role: Mapped[str] = mapped_column(String(20), default="employee")
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


Base.metadata.create_all(engine)
app = FastAPI(title="Enterprise RAG Assistant")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=8)


class LoginIn(RegisterIn):
    pass


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = None


class RoleIn(BaseModel):
    role: Literal["employee", "manager", "admin"]


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def issue_token(user: User) -> str:
    return jwt.encode({"sub": str(user.id), "role": user.role, "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)}, SECRET_KEY, algorithm=ALGORITHM)


def current_user(request: Request, db: Session = Depends(db_session)) -> User:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Sign in required")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        user = None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_role(*roles: str):
    def guard(user: User = Depends(current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return guard


_model = None
def embed(texts: list[str]) -> np.ndarray:
    """Load an efficient local transformer on first use; deterministic fallback keeps the demo offline."""
    global _model
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    except Exception:
        vectors = []
        for text in texts:
            v = np.zeros(384, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 384] += 1
            norm = np.linalg.norm(v)
            vectors.append(v / norm if norm else v)
        return np.asarray(vectors)


def split_text(text: str, size: int = 900, overlap: int = 160) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/auth/register")
def register(data: RegisterIn, db: Session = Depends(db_session)):
    email = data.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists")
    role = "admin" if db.scalar(select(User.id).limit(1)) is None else "employee"
    user = User(email=email, password_hash=pwd_context.hash(data.password), role=role)
    db.add(user); db.commit(); db.refresh(user)
    return {"token": issue_token(user), "user": {"id": user.id, "email": user.email, "role": user.role}}


@app.post("/api/auth/login")
def login(data: LoginIn, db: Session = Depends(db_session)):
    user = db.scalar(select(User).where(User.email == data.email.strip().lower()))
    if not user or not pwd_context.verify(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {"token": issue_token(user), "user": {"id": user.id, "email": user.email, "role": user.role}}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


@app.post("/api/documents")
async def upload_document(access_role: Literal["employee", "manager", "admin"], file: UploadFile = File(...), user: User = Depends(require_role("admin", "manager")), db: Session = Depends(db_session)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "PDF must be smaller than 25 MB")
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw))
        pages = [(n + 1, page.extract_text() or "") for n, page in enumerate(reader.pages)]
    except Exception as exc:
        raise HTTPException(400, f"Could not read PDF: {exc}")
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}.pdf"
    (UPLOADS / safe_name).write_bytes(raw)
    doc = Document(id=doc_id, title=Path(file.filename).stem, filename=safe_name, access_role=access_role, uploaded_by=user.id)
    db.add(doc)
    pending = [(page, part) for page, text in pages for part in split_text(text)]
    if not pending:
        raise HTTPException(400, "No readable text was found in this PDF")
    vectors = embed([p[1] for p in pending])
    for (page, content), vector in zip(pending, vectors):
        db.add(Chunk(document_id=doc_id, page=page, content=content, embedding=vector.tobytes()))
    db.commit()
    return {"id": doc_id, "title": doc.title, "chunks": len(pending), "message": "Document indexed successfully"}


@app.get("/api/documents")
def documents(user: User = Depends(current_user), db: Session = Depends(db_session)):
    rows = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [{"id": d.id, "title": d.title, "access_role": d.access_role, "created_at": d.created_at.isoformat(), "visible": ROLES[user.role] >= ROLES[d.access_role]} for d in rows]


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str, _: User = Depends(require_role("admin")), db: Session = Depends(db_session)):
    """Permanently delete the source PDF and the chunks embedded from it."""
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    stored_file = UPLOADS / document.filename
    db.delete(document)  # The relationship cascade deletes every indexed Chunk.
    db.commit()
    stored_file.unlink(missing_ok=True)
    return {"message": "Document and indexed chunks deleted"}


@app.get("/api/conversations")
def conversations(user: User = Depends(current_user), db: Session = Depends(db_session)):
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in db.scalars(select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc())).all()]


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)):
    c = db.get(Conversation, conversation_id)
    if not c or c.user_id != user.id: raise HTTPException(404, "Conversation not found")
    return [{"role": m.role, "content": m.content, "citations": m.citations} for m in c.messages]


@app.post("/api/chat")
def chat(data: ChatIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    c = db.get(Conversation, data.conversation_id) if data.conversation_id else None
    if c and c.user_id != user.id: raise HTTPException(403, "Conversation belongs to another user")
    if not c:
        c = Conversation(id=str(uuid.uuid4()), user_id=user.id, title=data.question[:70])
        db.add(c); db.flush()
    visible = db.scalars(select(Chunk).join(Document).where(Document.access_role.in_([r for r, rank in ROLES.items() if rank <= ROLES[user.role]]))).all()
    if not visible:
        answer, citations = "I couldn't find any documents you are allowed to access. Ask an administrator to upload one.", []
    else:
        q = embed([data.question])[0]
        ranked = sorted(((float(np.dot(q, np.frombuffer(x.embedding, dtype=np.float32))), x) for x in visible), key=lambda item: item[0], reverse=True)[:4]
        relevant = [(score, x) for score, x in ranked if score > 0.12]
        if not relevant:
            answer, citations = "I couldn't find an answer in the documents available to you.", []
        else:
            citations = [{"document": x.document.title, "page": x.page, "excerpt": x.content[:280]} for _, x in relevant]
            evidence = "\n\n".join(f"[{i + 1}] {x.content}" for i, (_, x) in enumerate(relevant))
            answer = "Based on the available documents:\n\n" + evidence[:3200] + "\n\nUse the cited sources below to verify the details."
    db.add(Message(conversation_id=c.id, role="user", content=data.question))
    db.add(Message(conversation_id=c.id, role="assistant", content=answer, citations=__import__("json").dumps(citations)))
    db.commit()
    return {"conversation_id": c.id, "answer": answer, "citations": citations}


@app.get("/api/admin/users")
def users(_: User = Depends(require_role("admin")), db: Session = Depends(db_session)):
    return [{"id": u.id, "email": u.email, "role": u.role, "created_at": u.created_at.isoformat()} for u in db.scalars(select(User).order_by(User.created_at)).all()]


@app.patch("/api/admin/users/{user_id}")
def update_user(user_id: int, data: RoleIn, _: User = Depends(require_role("admin")), db: Session = Depends(db_session)):
    target = db.get(User, user_id)
    if not target: raise HTTPException(404, "User not found")
    target.role = data.role; db.commit()
    return {"id": target.id, "role": target.role}


@app.get("/health")
def health(): return {"status": "ok"}
