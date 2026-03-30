import pytest


class TestBooksAPI:
    def test_create_book(self, client, sample_book_data):
        response = client.post("/api/books/", json=sample_book_data)
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == sample_book_data["title"]
        assert data["author"] == sample_book_data["author"]
        assert data["rating"] == 5
        assert "id" in data

    def test_create_book_minimal(self, client):
        response = client.post("/api/books/", json={
            "title": "Minimal Book",
            "author": "Test Author",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Minimal Book"
        assert data["rating"] is None
        assert data["read"] is False

    def test_get_books(self, client, sample_book_data):
        client.post("/api/books/", json=sample_book_data)
        response = client.get("/api/books/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == sample_book_data["title"]

    def test_get_books_empty(self, client):
        response = client.get("/api/books/")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_book_by_id(self, client, sample_book_data):
        create_response = client.post("/api/books/", json=sample_book_data)
        book_id = create_response.json()["id"]
        response = client.get(f"/api/books/{book_id}")
        assert response.status_code == 200
        assert response.json()["id"] == book_id

    def test_get_book_not_found(self, client):
        response = client.get("/api/books/999")
        assert response.status_code == 404

    def test_update_book(self, client, sample_book_data):
        create_response = client.post("/api/books/", json=sample_book_data)
        book_id = create_response.json()["id"]
        
        response = client.put(f"/api/books/{book_id}", json={
            "title": "Updated Title",
            "rating": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["rating"] == 3
        assert data["author"] == sample_book_data["author"]

    def test_delete_book(self, client, sample_book_data):
        create_response = client.post("/api/books/", json=sample_book_data)
        book_id = create_response.json()["id"]
        
        response = client.delete(f"/api/books/{book_id}")
        assert response.status_code == 204
        
        get_response = client.get(f"/api/books/{book_id}")
        assert get_response.status_code == 404

    def test_invalid_rating_too_high(self, client):
        response = client.post("/api/books/", json={
            "title": "Test",
            "author": "Author",
            "rating": 6,
        })
        assert response.status_code == 422

    def test_invalid_rating_too_low(self, client):
        response = client.post("/api/books/", json={
            "title": "Test",
            "author": "Author",
            "rating": 0,
        })
        assert response.status_code == 422

    def test_rating_validation(self, client):
        for rating in range(1, 6):
            response = client.post("/api/books/", json={
                "title": f"Book {rating}",
                "author": "Author",
                "rating": rating,
            })
            assert response.status_code == 201
            assert response.json()["rating"] == rating
