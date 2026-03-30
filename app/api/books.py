import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.book import Book
from app.schemas.book import BookCreate, BookUpdate, Book, OcrResult, BatchScanResponse
from app.services.ocr import OcrService

router = APIRouter(prefix="/api/books", tags=["books"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_model=List[Book])
def get_books(db: Session = Depends(get_db)):
    books = db.query(Book).all()
    return books


@router.get("/{book_id}", response_model=Book)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@router.post("/", response_model=Book, status_code=status.HTTP_201_CREATED)
def create_book(book: BookCreate, db: Session = Depends(get_db)):
    db_book = Book(**book.model_dump())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book


@router.put("/{book_id}", response_model=Book)
def update_book(book_id: int, book: BookUpdate, db: Session = Depends(get_db)):
    db_book = db.query(Book).filter(Book.id == book_id).first()
    if not db_book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    for key, value in book.model_dump(exclude_unset=True).items():
        setattr(db_book, key, value)
    
    db.commit()
    db.refresh(db_book)
    return db_book


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    db_book = db.query(Book).filter(Book.id == book_id).first()
    if not db_book:
        raise HTTPException(status_code=404, detail="Book not found")
    db.delete(db_book)
    db.commit()


@router.post("/scan", response_model=BatchScanResponse)
async def scan_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_paths = []
    for upload_file in files:
        if upload_file.filename:
            ext = os.path.splitext(upload_file.filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                filename = f"{uuid.uuid4()}{ext}"
                filepath = os.path.join(upload_dir, filename)
                with open(filepath, "wb") as f:
                    content = await upload_file.read()
                    f.write(content)
                image_paths.append(filepath)

    if not image_paths:
        raise HTTPException(status_code=400, detail="No valid image files provided")

    def process_images():
        return OcrService.process_batch(image_paths)

    result = await background_tasks.run_in_executor(None, process_images)
    return result


@router.post("/import", response_model=List[Book])
def import_books(
    books: List[OcrResult],
    db: Session = Depends(get_db)
):
    imported = []
    for book_data in books:
        if book_data.title and book_data.author:
            db_book = Book(
                title=book_data.title,
                author=book_data.author,
                isbn=book_data.isbn,
                needs_review=book_data.needs_review
            )
            db.add(db_book)
            imported.append(db_book)

    db.commit()
    for book in imported:
        db.refresh(book)

    return imported


@router.get("/review", response_model=List[Book])
def get_books_needing_review(db: Session = Depends(get_db)):
    books = db.query(Book).filter(Book.needs_review == True).all()
    return books
