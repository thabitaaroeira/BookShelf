import os
import uuid
from fastapi import APIRouter, Depends, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.book import Book
from app.services.ocr import OcrService

templates = Jinja2Templates(directory="app/templates")
router = APIRouter(prefix="/books", tags=["web"])


@router.get("/", response_class=HTMLResponse)
def list_books(request: Request, db: Session = Depends(get_db)):
    books = db.query(Book).order_by(Book.id.desc()).all()
    return templates.TemplateResponse("index.html", {"request": request, "books": books})


@router.get("/new", response_class=HTMLResponse)
def new_book_form(request: Request):
    return templates.TemplateResponse("partials/book_form.html", {"request": request})


@router.get("/{book_id}/edit", response_class=HTMLResponse)
def edit_book_form(request: Request, book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return templates.TemplateResponse("partials/book_edit.html", {"request": request, "book": book})


@router.get("/close-modal", response_class=HTMLResponse)
def close_modal():
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
    if rating is not None and (rating < 1 or rating > 5):
        rating = None
    book = Book(title=title, author=author, isbn=isbn, read=read, rating=rating, notes=notes)
    db.add(book)
    db.commit()
    db.refresh(book)
    
    books = db.query(Book).order_by(Book.id.desc()).all()
    return templates.TemplateResponse("partials/book_list.html", {"request": request, "books": books})


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
    return templates.TemplateResponse("partials/book_list.html", {"request": request, "books": books})


@router.delete("/{book_id}", response_class=HTMLResponse)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    db.delete(book)
    db.commit()
    return ""


@router.get("/scan", response_class=HTMLResponse)
def scan_page(request: Request):
    return templates.TemplateResponse("partials/scan_form.html", {"request": request})


@router.post("/scan", response_class=HTMLResponse)
async def process_scan(
    request: Request,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
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
                content = await upload_file.read()
                with open(filepath, "wb") as f:
                    f.write(content)
                image_paths.append(filepath)

    if not image_paths:
        return templates.TemplateResponse(
            "partials/scan_results.html",
            {"request": request, "error": "No valid image files provided", "results": [], "needs_review": []}
        )

    result = OcrService.process_batch(image_paths)

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

    return templates.TemplateResponse(
        "partials/scan_results.html",
        {"request": request, "results": result.successful, "needs_review": result.needs_review, "total": result.total_processed, "imported": len(imported_books)}
    )


@router.get("/review", response_class=HTMLResponse)
def review_page(request: Request, db: Session = Depends(get_db)):
    books = db.query(Book).filter(Book.needs_review == True).order_by(Book.id.desc()).all()
    return templates.TemplateResponse("partials/review_list.html", {"request": request, "books": books})
