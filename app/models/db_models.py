"""
SQLAlchemy ORM models for MedGuard RAG.

Three tables, deliberately normalized:
  - Drug            : one row per unique drug
  - Interaction      : one row per drug-pair interaction (structured, exact-match source of truth)
  - LabelChunk       : one row per chunk of unstructured FDA label text (the RAG retrieval corpus)

Design note: we do NOT store embedding vectors here. Postgres holds text + metadata;
Qdrant holds the actual vectors. `LabelChunk.vector_id` is the join key between the two.
This mirrors how production RAG systems separate "system of record" from "search index".
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    pass


class Severity(str, enum.Enum):
    """
    Mirrors DDInter's risk-level taxonomy.
    Kept as an explicit enum (not a free-text string) so the retrieval and
    generation layers can reason about severity programmatically -- e.g.
    "always surface Major interactions first, regardless of vector similarity rank."
    """

    MINOR = "Minor"
    MODERATE = "Moderate"
    MAJOR = "Major"
    UNKNOWN = "Unknown"


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String(255), nullable=False, unique=True, index=True)
    # ATC = Anatomical Therapeutic Chemical code, e.g. "B01AA03" for warfarin.
    # Nullable because not every drug in a CSV will have one populated.
    atc_code: Mapped[str | None] = Column(String(16), nullable=True, index=True)
    drug_class: Mapped[str | None] = Column(String(128), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())

    label_chunks: Mapped[list["LabelChunk"]] = relationship(
        back_populates="drug", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Drug id={self.id} name={self.name!r}>"


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        # Prevent the same unordered pair being loaded twice (A,B) and (B,A)
        # as separate rows -- we normalize pair order at ingestion time instead.
        UniqueConstraint("drug_a_id", "drug_b_id", name="uq_interaction_pair"),
    )

    id: Mapped[int] = Column(Integer, primary_key=True)
    drug_a_id: Mapped[int] = Column(ForeignKey("drugs.id"), nullable=False, index=True)
    drug_b_id: Mapped[int] = Column(ForeignKey("drugs.id"), nullable=False, index=True)
    severity: Mapped[Severity] = Column(Enum(Severity), nullable=False, default=Severity.UNKNOWN)
    mechanism: Mapped[str | None] = Column(Text, nullable=True)
    management: Mapped[str | None] = Column(Text, nullable=True)
    source: Mapped[str] = Column(String(64), nullable=False, default="ddinter")
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())

    drug_a: Mapped["Drug"] = relationship(foreign_keys=[drug_a_id])
    drug_b: Mapped["Drug"] = relationship(foreign_keys=[drug_b_id])

    def __repr__(self) -> str:
        return f"<Interaction {self.drug_a_id}-{self.drug_b_id} severity={self.severity}>"


class LabelChunk(Base):
    """
    A single retrievable chunk of unstructured clinical text (FDA label section,
    warning, dosage guidance, etc). This table is the bridge between Postgres
    (source of truth for text + metadata) and Qdrant (vector index for similarity search).
    """

    __tablename__ = "label_chunks"

    id: Mapped[int] = Column(Integer, primary_key=True)
    drug_id: Mapped[int] = Column(ForeignKey("drugs.id"), nullable=False, index=True)
    chunk_text: Mapped[str] = Column(Text, nullable=False)
    # e.g. "boxed_warning", "dosage_and_administration", "contraindications"
    section: Mapped[str] = Column(String(128), nullable=False, index=True)
    chunk_index: Mapped[int] = Column(Integer, nullable=False, default=0)
    # Points to the corresponding point ID in the Qdrant collection.
    vector_id: Mapped[str | None] = Column(String(64), nullable=True, unique=True, index=True)
    source_url: Mapped[str | None] = Column(String(512), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())

    drug: Mapped["Drug"] = relationship(back_populates="label_chunks")

    def __repr__(self) -> str:
        return f"<LabelChunk id={self.id} drug_id={self.drug_id} section={self.section!r}>"
