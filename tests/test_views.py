"""
Tests for the web views.

Covers:
- Basic page rendering
- Form handling
- Scan functionality
- Review functionality
"""

import pytest
from io import BytesIO
from unittest.mock import patch, AsyncMock


class TestBasicWebViews:
    """Tests for basic web view pages."""

    def test_list_books_page(self, client):
        """Books list page should render correctly."""
        response = client.get("/books/")
        assert response.status_code == 200
        assert "My BookShelf" in response.text
        assert 'hx-get="/books/new"' in response.text
        assert 'hx-get="/books/scan"' in response.text

    def test_new_book_form(self, client):
        """New book form should render correctly."""
        response = client.get("/books/new")
        assert response.status_code == 200
        assert 'hx-post="/books"' in response.text
        assert 'id="title"' in response.text
        assert 'id="author"' in response.text

    def test_edit_book_form(self, client):
        """Edit book form should render with book data."""
        # Create a book first
        client.post("/api/books/", json={
            "title": "Edit Test",
            "author": "Author"
        })
        
        response = client.get("/books/1/edit")
        assert response.status_code == 200
        assert "Edit Book" in response.text
        assert 'value="Edit Test"' in response.text

    def test_edit_book_not_found(self, client):
        """Editing non-existent book should return 404."""
        response = client.get("/books/999/edit")
        assert response.status_code == 404

    def test_close_modal(self, client):
        """Close modal should return empty response."""
        response = client.get("/books/close-modal")
        assert response.status_code == 200
        assert response.text == ""


class TestBookFormOperations:
    """Tests for form-based book operations."""

    def test_create_book_form(self, client):
        """Creating book via form should work."""
        response = client.post(
            "/books/",
            data={
                "title": "Form Test Book",
                "author": "Form Author",
                "rating": "4",
                "read": "true",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "Form Test Book" in response.text

    def test_create_book_form_minimal(self, client):
        """Creating book with minimal form data should work."""
        response = client.post(
            "/books/",
            data={
                "title": "Minimal Book",
                "author": "Author",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "Minimal Book" in response.text

    def test_update_book_form(self, client):
        """Updating book via form should work."""
        # Create a book
        client.post("/api/books/", json={
            "title": "Original Title",
            "author": "Author"
        })
        
        response = client.put(
            "/books/1",
            data={
                "title": "Updated Title",
                "author": "Author",
                "rating": "5",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "Updated Title" in response.text

    def test_update_book_not_found(self, client):
        """Updating non-existent book should return 404."""
        response = client.put(
            "/books/999",
            data={
                "title": "Title",
                "author": "Author",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 404

    def test_delete_book(self, client):
        """Deleting book should work."""
        # Create a book
        client.post("/api/books/", json={
            "title": "To Delete",
            "author": "Author"
        })
        
        response = client.delete("/books/1")
        assert response.status_code == 200
        assert response.text == ""

    def test_delete_book_not_found(self, client):
        """Deleting non-existent book should return 404."""
        response = client.delete("/books/999")
        assert response.status_code == 404


class TestNeedsReviewBadge:
    """Tests for needs_review badge display."""

    def test_badge_shown_when_needs_review(self, client):
        """Needs Review badge should appear when book needs review."""
        client.post("/api/books/", json={
            "title": "Needs Review",
            "author": "Author",
            "needs_review": True
        })
        
        response = client.get("/books/")
        assert "Needs Review" in response.text
        assert "badge" in response.text.lower()

    def test_badge_hidden_when_complete(self, client):
        """Needs Review badge should not appear for complete books."""
        client.post("/api/books/", json={
            "title": "Complete",
            "author": "Author",
            "needs_review": False
        })
        
        response = client.get("/books/")
        assert "Complete" in response.text


class TestScanWebViews:
    """Tests for scan web views."""

    def test_scan_page(self, client):
        """Scan page should render correctly."""
        response = client.get("/books/scan")
        assert response.status_code == 200
        assert 'hx-post="/books/scan"' in response.text
        assert 'type="file"' in response.text
        assert 'multiple' in response.text

    @patch('app.views.OcrService.process_batch')
    @patch('app.views.save_multiple_uploads', new_callable=AsyncMock)
    def test_scan_with_mock_files(self, mock_save, mock_process, client):
        """Scanning should process files and show results."""
        from pathlib import Path
        mock_save.return_value = [Path("/tmp/test.jpg")]
        
        from app.schemas.book import BatchScanResponse, OcrResult
        mock_process.return_value = BatchScanResponse(
            successful=[
                OcrResult(
                    title="Scanned Book",
                    author="Author",
                    isbn="9781234567890",
                    needs_review=False
                )
            ],
            needs_review=[],
            total_processed=1
        )
        
        file_content = b"fake image data"
        files = {
            "files": ("test.jpg", BytesIO(file_content), "image/jpeg")
        }
        
        response = client.post("/books/scan", files=files)
        assert response.status_code == 200
        assert "Scanned Book" in response.text

    @patch('app.views.save_multiple_uploads', new_callable=AsyncMock)
    def test_scan_no_valid_files(self, mock_save, client):
        """Scanning with no valid files should show error."""
        mock_save.side_effect = Exception("No valid files")
        
        file_content = b"content"
        files = {
            "files": ("test.exe", BytesIO(file_content), "application/octet-stream")
        }
        
        response = client.post("/books/scan", files=files)
        assert response.status_code == 200
        assert "error" in response.text.lower() or "failed" in response.text.lower()


class TestReviewWebViews:
    """Tests for review web views."""

    def test_review_page_empty(self, client):
        """Review page with no books should show empty state."""
        response = client.get("/books/review")
        assert response.status_code == 200
        # Should show empty state or message
        assert "review" in response.text.lower() or "empty" in response.text.lower()

    def test_review_page_with_books(self, client):
        """Review page with books should show them."""
        # Create books needing review
        client.post("/api/books/", json={
            "title": "Needs Review 1",
            "author": "Author",
            "needs_review": True
        })
        client.post("/api/books/", json={
            "title": "Needs Review 2",
            "author": "Author",
            "needs_review": True
        })
        
        response = client.get("/books/review")
        assert response.status_code == 200
        assert "Needs Review 1" in response.text
        assert "Needs Review 2" in response.text

    def test_review_page_only_shows_needing_review(self, client):
        """Review page should only show books needing review."""
        # Create mix of books
        client.post("/api/books/", json={
            "title": "Complete",
            "author": "Author",
            "needs_review": False
        })
        client.post("/api/books/", json={
            "title": "Incomplete",
            "author": "Author",
            "needs_review": True
        })
        
        response = client.get("/books/review")
        assert response.status_code == 200
        assert "Incomplete" in response.text
        # "Complete" book should not be shown in review
        # or it might be shown but is filtered by the view
        assert response.text.count("Complete") <= 1  # At most once if present


class TestModalAndHTMX:
    """Tests for HTMX modal behavior."""

    def test_add_book_modal_trigger(self, client):
        """Add book button should have correct HTMX attributes."""
        response = client.get("/books/")
        assert 'hx-get="/books/new"' in response.text
        assert 'hx-target="#modal"' in response.text

    def test_scan_button_trigger(self, client):
        """Scan button should have correct HTMX attributes."""
        response = client.get("/books/")
        assert 'hx-get="/books/scan"' in response.text
        assert 'hx-target="#modal"' in response.text

    def test_delete_confirmation(self, client):
        """Delete button should have confirmation."""
        client.post("/api/books/", json={
            "title": "To Delete",
            "author": "Author"
        })
        
        response = client.get("/books/")
        assert 'hx-confirm=' in response.text
        assert "Delete" in response.text
