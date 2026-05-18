"""User credential ORM model — encrypted per-user API keys."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK

if TYPE_CHECKING:
    from pgw.db.models.user import User


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserCredential(Base):
    __tablename__ = "user_credentials"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    service: Mapped[str] = mapped_column(String(20), nullable=False)  # "whisper" | "llm"
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # "groq" | "openai" | …
    encrypted_value: Mapped[str] = mapped_column(Text, nullable=False)  # AES-256-GCM ciphertext
    api_base: Mapped[str | None] = mapped_column(String(500), nullable=True)
    api_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    user: Mapped["User"] = relationship(back_populates="credentials")
