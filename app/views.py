"""
Web view routes for book operations.

Provides HTML view routes for:
- Listing books
- Creating/editing books via forms
- Scanning book spines
- Reviewing incomplete OCR results
"""

import logging
from pathlib import Path
from fastapi import APIRouter, Depends, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.book import Book
from app.services.ocr import OcrService
from app.services.upload import save_multiple_uploads, UploadError


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/books", tags=["web"])


@router.get("/", response_class=HTMLResponse)
def list_books(request: Request, db: Session = Depends(get_db)):
    """Display all books."""
    books = db.query(Book).order_by(Book.id.desc()).all()
    return _render("index.html", request, {"books": books})


@router.get("/new", response_class=HTMLResponse)
def new_book_form(request: Request):
    """Show form for adding a new book."""
    return _render("partials/book_form.html", request)


@router.get("/{book_id}/edit", response_class=HTMLResponse)
def edit_book_form(request: Request, book_id: int, db: Session = Depends(get_db)):
    """Show form for editing a book."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return _render("partials/book_edit.html", request, {"book": book})


@router.get("/close-modal", response_class=HTMLResponse)
def close_modal():
    """Close the modal dialog."""
    return ""


@router.post("/", response_class=HTMLResponse)
def create_book(
    request: Request,
    title: str = Form(...),
    author: str = Form(...),
    isbn: str = Form(None),
    read: bool = Form(False),
    rating: int = Form(None),
    notes: str = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new book from form data."""
    if rating is not None and (rating < 1 or rating > 5):
        rating = None
    
    book = Book(
        title=title,
        author=author,
        isbn=isbn,
        read=read,
        rating=rating,
        notes=notes
    )
    db.add(book)
    db.commit()
    db.refresh(book)
    
    books = db.query(Book).order_by(Book.id.desc()).all()
    return _render("partials/book_list.html", request, {"books": books})


@router.put("/{book_id}", response_class=HTMLResponse)
def update_book(
    request: Request,
    book_id: int,
    title: str = Form(...),
    author: str = Form(...),
    isbn: str = Form(None),
    read: bool = Form(False),
    rating: int = Form(None),
    notes: str = Form(None),
    db: Session = Depends(get_db)
):
    """Update an existing book from form data."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    if rating is not None and (rating < 1 or rating > 5):
        rating = None
    
    book.title = title
    book.author = author
    book.isbn = isbn
    book.read = read
    book.rating = rating
    book.notes = notes
    book.needs_review = False
    db.commit()
    
    books = db.query(Book).order_by(Book.id.desc()).all()
    return _render("partials/book_list.html", request, {"books": books})


@router.delete("/{book_id}", response_class=HTMLResponse)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    """Delete a book."""
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    db.delete(book)
    db.commit()
    return ""


@router.get("/scan", response_class=HTMLResponse)
def scan_page(request: Request):
    """Show the scan books page."""
    return _render("partials/scan_form.html", request)


@router.post("/scan", response_class=HTMLResponse)
async def process_scan(
    request: Request,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Process uploaded book spine images.
    
    Saves files securely, runs OCR, creates book records,
    and returns results showing successful imports and
    items needing review.
    """
    try:
        image_paths = await save_multiple_uploads(files)
    except UploadError as e:
        return _render("partials/scan_results.html", request, {
            "error": str(e),
            "results": [],
            "needs_review": []
        })
    
    if not image_paths:
        return _render("partials/scan_results.html", request, {
            "error": "No valid image files provided",
            "results": [],
            "needs_review": []
        })
    
    result = OcrService.process_batch([str(p) for p in image_paths])
    
    imported_books = []
    for ocr_result in result.successful + result.needs_review:
        if ocr_result.title and ocr_result.author:
            book = Book(
                title=ocr_result.title,
                author=ocr_result.author,
                isbn=ocr_result.isbn,
                needs_review=ocr_result.needs_review
            )
            db.add(book)
            imported_books.append(book)
    
    db.commit()
    for book in imported_books:
        db.refresh(book)
    
    logger.info(
        f"Scan complete: {len(result.successful)} successful, "
        f"{len(result.needs_review)} need review"
    )
    
    return _render("partials/scan_results.html", request, {
        "results": result.successful,
        "needs_review": result.needs_review,
        "total": result.total_processed,
        "imported": len(imported_books)
    })


@router.get("/review", response_class=HTMLResponse)
def review_page(request: Request, db: Session = Depends(get_db)):
    """Show books that need manual review."""
    books = db.query(Book).filter(Book.needs_review == True).order_by(Book.id.desc()).all()
    return _render("partials/review_list.html", request, {"books": books})


def _render(template: str, request: Request, context: dict | None = None) -> HTMLResponse:
    """Helper to render templates with consistent context."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="app/templates")
    ctx: dict = {"request": request}
    if context:
        ctx.update(context)
    return templates.TemplateResponse(request, template, ctx)
