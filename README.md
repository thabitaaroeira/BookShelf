# BookShelf

A personal book library built with **FastAPI + Jinja2 + HTMX**.

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Backend | FastAPI | REST API + web routes |
| ORM | SQLAlchemy | Database operations |
| Database | SQLite | Persistent storage |
| Templates | Jinja2 | Server-side HTML rendering |
| Frontend | HTMX | Dynamic interactions (no JS needed) |

## Quick Start

```bash
cd BookShelf
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## URLs

| URL | Description |
|-----|-------------|
| <http://localhost:8000> | BookShelf web app |
| <http://localhost:8000/docs> | Swagger API docs |
| <http://localhost:8000/redoc> | ReDoc API docs |

## Features

- Add, edit, delete books
- Track read/unread status
- 1-5 star rating system
- Add notes to books
- ISBN support
- No page reloads (HTMX)

## Testing

```bash
cd BookShelf
pip install -r requirements.txt
pytest -v
```

Run with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py          # Pytest fixtures
├── test_api.py          # API endpoint tests
└── test_views.py        # Web view tests
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/books | List all books |
| GET | /api/books/{id} | Get book by ID |
| POST | /api/books | Create new book |
| PUT | /api/books/{id} | Update book |
| DELETE | /api/books/{id} | Delete book |

## Project Structure

```
BookShelf/
├── requirements.txt
├── README.md
└── app/
    ├── main.py           # FastAPI app
    ├── database.py       # SQLite setup
    ├── views.py          # Web UI routes
    ├── api/
    │   └── books.py      # REST API
    ├── models/
    │   └── book.py       # SQLAlchemy model
    ├── schemas/
    │   └── book.py       # Pydantic schemas
    └── templates/
        ├── base.html     # Layout + HTMX
        ├── index.html    # Main page
        └── partials/
            ├── book_list.html
            ├── book_form.html
            └── book_edit.html
```

## What's HTMX Doing Here?

HTMX provides AJAX-like behavior without writing JavaScript:

- `hx-get` - fetch content on click
- `hx-post` - submit forms via AJAX
- `hx-target` - specify where to put the response
- `hx-swap` - how to swap the content (innerHTML, outerHTML)
- `hx-swap-oob` - swap content into a different element (modal)

The form modal is rendered as a partial and swapped into the `#modal` div via `hx-swap-oob`.
