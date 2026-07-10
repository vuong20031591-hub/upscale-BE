"""Declarative base for ORM models."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base."""
    pass
