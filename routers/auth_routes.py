from typing import List, Optional
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models_db import User
from auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    require_role,
)

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "student"  # admin, faculty, student


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact administrator.",
        )

    token = create_access_token(data={"sub": user.email, "role": user.role})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        email=user.email,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/users", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    users = db.query(User).order_by(User.id.asc()).all()
    return users


@router.post("/users", response_model=UserResponse)
def create_user(
    req: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if req.role not in ["admin", "faculty", "student"]:
        raise HTTPException(status_code=400, detail="Role must be admin, faculty, or student")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User with email '{email}' already exists")

    new_user = User(
        email=email,
        password_hash=get_password_hash(req.password),
        role=req.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
