import pytest


class TestWebViews:
    def test_list_books_page(self, client):
        response = client.get("/books/")
        assert response.status_code == 200
        assert "My BookShelf" in response.text

    def test_new_book_form(self, client):
        response = client.get("/books/new")
        assert response.status_code == 200
        assert 'hx-post="/books"' in response.text
        assert 'id="title"' in response.text

    def test_create_book_form(self, client):
        response = client.post(
            "/books/",
            data={
                "title": "Form Test Book",
                "author": "Form Author",
                "rating": "4",
                "read": True,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "Form Test Book" in response.text
        assert "★★★★" in response.text

    def test_edit_book_form(self, client):
        create_response = client.post(
            "/api/books/",
            json={"title": "Edit Test", "author": "Author"},
        )
        book_id = create_response.json()["id"]
        
        response = client.get(f"/books/{book_id}/edit")
        assert response.status_code == 200
        assert "Edit Book" in response.text
        assert 'value="Edit Test"' in response.text

    def test_update_book_form(self, client):
        create_response = client.post(
            "/api/books/",
            json={"title": "Update Test", "author": "Author"},
        )
        book_id = create_response.json()["id"]
        
        response = client.put(
            f"/books/{book_id}",
            data={
                "title": "Updated Title",
                "author": "Author",
                "rating": "5",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "Updated Title" in response.text

    def test_delete_book(self, client):
        create_response = client.post(
            "/api/books/",
            json={"title": "Delete Test", "author": "Author"},
        )
        book_id = create_response.json()["id"]
        
        response = client.delete(f"/books/{book_id}")
        assert response.status_code == 200
        assert response.text == ""

    def test_close_modal(self, client):
        response = client.get("/books/close-modal")
        assert response.status_code == 200
        assert response.text == ""
