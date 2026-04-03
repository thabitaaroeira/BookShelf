"""
API routes for book operations.

Provides REST API endpoints for:
- Listing, creating, reading, updating, deleting books
- Scanning book spines via OCR
- Importing books from OCR results
- Retrieving books needing review
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.book import Book as BookModel
from app.schemas.book import BookCreate, BookUpdate, Book, OcrResult, BatchScanResponse
from app.services.ocr import OcrService
from app.services.upload import save_multiple_uploads, UploadError


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/books", tags=["books"])


@router.get("/", response_model=List[Book])
def list_books(db: Session = Depends(get_db)):
    """Get all books."""
    return db.query(BookModel).all()


@router.get("/{book_id}", response_model=Book)
def get_book(book_id: int, db: Session = Depends(get_db)):
    """Get a single book by ID."""
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@router.post("/", response_model=Book, status_code=status.HTTP_201_CREATED)
def create_book(book_data: BookCreate, db: Session = Depends(get_db)):
    """Create a new book."""
    book = BookModel(**book_data.model_dump())
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


@router.put("/{book_id}", response_model=Book)
def update_book(
    book_id: int,
    book_data: BookUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing book."""
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    for key, value in book_data.model_dump(exclude_unset=True).items():
        setattr(book, key, value)
    
    db.commit()
    db.refresh(book)
    return book


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    """Delete a book."""
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    db.delete(book)
    db.commit()


@router.post("/scan", response_model=BatchScanResponse)
async def scan_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Scan book spine images and extract book information.
    
    Accepts multiple image files, performs OCR on each,
    and returns results categorized by success/needs_review.
    """
    try:
        image_paths = await save_multiple_uploads(files)
    except UploadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not image_paths:
        raise HTTPException(status_code=400, detail="No valid image files provided")
    
    def process_images():
        return OcrService.process_batch([str(p) for p in image_paths])
    
    result = await background_tasks.run_in_executor(None, process_images)
    return result


@router.post("/import", response_model=List[Book], status_code=status.HTTP_201_CREATED)
def import_books(
    books: List[OcrResult],
    db: Session = Depends(get_db)
):
    """
    Import books from OCR results.
    
    Creates book records from a list of OCR results,
    only importing those with both title and author.
    """
    imported = []
    for book_data in books:
        if book_data.title and book_data.author:
            book = BookModel(
                title=book_data.title,
                author=book_data.author,
                isbn=book_data.isbn,
                needs_review=book_data.needs_review
            )
            db.add(book)
            imported.append(book)
    
    db.commit()
    for book in imported:
        db.refresh(book)
    
    logger.info(f"Imported {len(imported)} books from OCR results")
    return imported


@router.get("/review", response_model=List[Book])
def get_books_needing_review(db: Session = Depends(get_db)):
    """Get all books that need manual review."""
    return db.query(BookModel).filter(BookModel.needs_review == True).all()
