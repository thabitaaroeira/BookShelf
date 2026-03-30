from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)
    isbn = Column(String, unique=True, nullable=True)
    read = Column(Boolean, default=False)
    rating = Column(Integer, nullable=True)
    notes = Column(String, nullable=True)
    needs_review = Column(Boolean, default=False)
