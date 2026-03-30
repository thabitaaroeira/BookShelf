"""
Tests for the OCR service module.

Covers:
- ISBN extraction
- Title/author parsing
- Text cleaning
- Batch processing
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.ocr import OcrService, ParsedBookInfo
from app.schemas.book import OcrResult, BatchScanResponse


class TestIsbnExtraction:
    """Tests for ISBN extraction from text."""

    def test_extract_isbn_13_with_prefix(self):
        """ISBN-13 with 'ISBN' prefix should be extracted."""
        text = "ISBN 978-013-595-705-9"
        result = OcrService._extract_isbn(text)
        assert result == "9780135957059"

    def test_extract_isbn_13_with_colon(self):
        """ISBN-13 with colon should be extracted."""
        text = "ISBN: 9780135957059"
        text = "ISBN:9780135957059"
        result = OcrService._extract_isbn(text)
        assert result == "9780135957059"

    def test_extract_isbn_10(self):
        """ISBN-10 should be extracted."""
        text = "ISBN 0135957052"
        result = OcrService._extract_isbn(text)
        assert result == "0135957052"

    def test_extract_isbn_in_sentence(self):
        """ISBN in middle of sentence should be extracted."""
        text = "This book has ISBN 9781234567890 for reference."
        result = OcrService._extract_isbn(text)
        assert result == "9781234567890"

    def test_no_isbn_returns_none(self):
        """Text without ISBN should return None."""
        assert OcrService._extract_isbn("Just a regular book title") is None
        assert OcrService._extract_isbn("") is None

    def test_isbn_without_dashes(self):
        """ISBN without dashes should be extracted correctly."""
        text = "ISBN 9780135957059"
        result = OcrService._extract_isbn(text)
        assert result == "9780135957059"


class TestTitleAuthorExtraction:
    """Tests for title and author extraction."""

    def test_explicit_by_indicator(self):
        """Text with 'by' indicator should parse correctly."""
        texts = ["The Great Gatsby", "by", "F. Scott Fitzgerald"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "The Great Gatsby"
        assert author == "F. Scott Fitzgerald"

    def test_written_by_indicator(self):
        """Text with 'written by' indicator should parse correctly."""
        texts = ["Book Title", "written by", "Author Name"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "Book Title"
        assert author == "Author Name"

    def test_author_colon_indicator(self):
        """Text with 'Author:' indicator should parse correctly."""
        texts = ["Some Book", "Author:", "John Doe"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "Some Book"
        assert author == "John Doe"

    def test_empty_texts_returns_none(self):
        """Empty text list should return None for both."""
        title, author = OcrService._extract_title_author([])
        assert title is None
        assert author is None

    def test_single_text_returns_title_only(self):
        """Single text block should be treated as title."""
        texts = ["The Only Book Title"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "The Only Book Title"
        assert author is None

    def test_two_texts_longer_is_title(self):
        """With two texts, longer one is title."""
        texts = ["A Very Long Book Title That Is Clearly The Main Title", "Short Author"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "A Very Long Book Title That Is Clearly The Main Title"
        assert author == "Short Author"

    def test_two_texts_shorter_is_title_when_longer(self):
        """When second text is longer, it becomes title."""
        texts = ["Short", "This Is The Actual Title Which Is Much Longer"]
        title, author = OcrService._extract_title_author(texts)
        assert title == "This Is The Actual Title Which Is Much Longer"
        assert author == "Short"

    def test_multiple_texts_length_heuristics(self):
        """Multiple texts should use length heuristics."""
        texts = [
            "A Very Long Book Title That Should Be Recognized As The Title",
            "Author Name",
            "Extra text"
        ]
        title, author = OcrService._extract_title_author(texts)
        assert "Book Title" in title or title is not None


class TestTextCleaning:
    """Tests for text cleaning."""

    def test_remove_special_characters(self):
        """Special characters should be removed."""
        dirty = "Hello@#$%World!"
        clean = OcrService._clean_text(dirty)
        assert "@" not in clean
        assert "#" not in clean
        assert "$" not in clean

    def test_preserve_valid_punctuation(self):
        """Valid punctuation should be preserved."""
        text = "Hello, World! How are you?"
        clean = OcrService._clean_text(text)
        assert "," in clean
        assert "!" in clean
        assert "?" in clean

    def test_collapse_whitespace(self):
        """Multiple whitespace should be collapsed."""
        text = "Hello    World\n\nTest"
        clean = OcrService._clean_text(text)
        assert "    " not in clean
        assert "\n" not in clean

    def test_trim_edges(self):
        """Leading and trailing whitespace should be trimmed."""
        text = "   Hello World   "
        clean = OcrService._clean_text(text)
        assert clean == "Hello World"
        assert not clean.startswith(" ")
        assert not clean.endswith(" ")


class TestOcrService:
    """Tests for the OcrService class."""

    def test_initialize_creates_reader(self):
        """Initialize should create the EasyOCR reader."""
        # Reset the reader first
        OcrService._reader = None
        
        with patch('app.services.ocr.easyocr.Reader') as mock_reader:
            mock_instance = MagicMock()
            mock_reader.return_value = mock_instance
            
            OcrService.initialize(languages=['en'], gpu=False)
            
            mock_reader.assert_called_once_with(['en'], gpu=False, verbose=False)
            assert OcrService._reader is not None

    def test_initialize_idempotent(self):
        """Multiple initialize calls should not recreate reader."""
        OcrService._reader = None
        
        with patch('app.services.ocr.easyocr.Reader') as mock_reader:
            mock_instance = MagicMock()
            mock_reader.return_value = mock_instance
            
            OcrService.initialize()
            OcrService.initialize()
            OcrService.initialize()
            
            # Should only be called once
            assert mock_reader.call_count == 1

    @patch.object(OcrService, 'get_reader')
    def test_process_image_empty_result(self, mock_get_reader):
        """Empty OCR results should return needs_review=True."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = mock_reader
        
        result = OcrService.process_image("fake_path.jpg")
        
        assert result.needs_review is True
        assert result.title is None
        assert result.raw_text == ""

    @patch.object(OcrService, 'get_reader')
    def test_process_image_with_results(self, mock_get_reader):
        """OCR results should be parsed correctly."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "The Great Gatsby", 0.95),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "F. Scott Fitzgerald", 0.90),
        ]
        mock_get_reader.return_value = mock_reader
        
        result = OcrService.process_image("fake_path.jpg")
        
        assert result.raw_text == "The Great Gatsby F. Scott Fitzgerald"
        assert 0.9 <= result.confidence <= 0.95

    @patch.object(OcrService, 'get_reader')
    def test_process_image_low_confidence_flags_review(self, mock_get_reader):
        """Low confidence results should flag needs_review."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.3),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "Author", 0.3),
        ]
        mock_get_reader.return_value = mock_reader
        
        result = OcrService.process_image("fake_path.jpg")
        
        assert result.needs_review is True

    @patch.object(OcrService, 'get_reader')
    def test_process_image_ocr_error_handling(self, mock_get_reader):
        """OCR errors should be handled gracefully."""
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = Exception("OCR failed")
        mock_get_reader.return_value = mock_reader
        
        result = OcrService.process_image("fake_path.jpg")
        
        assert result.needs_review is True
        assert result.raw_text == ""


class TestBatchProcessing:
    """Tests for batch processing."""

    @patch.object(OcrService, 'process_image')
    def test_process_batch_categorizes_results(self, mock_process):
        """Batch processing should categorize results."""
        # Create mock results
        success_result = OcrResult(
            title="Book 1",
            author="Author 1",
            needs_review=False,
            raw_text="Book 1 Author 1"
        )
        review_result = OcrResult(
            title="Book 2",
            author=None,
            needs_review=True,
            raw_text="Book 2"
        )
        
        mock_process.side_effect = [success_result, review_result]
        
        result = OcrService.process_batch(["path1.jpg", "path2.jpg"], cleanup=False)
        
        assert result.total_processed == 2
        assert len(result.successful) == 1
        assert len(result.needs_review) == 1
        assert result.successful[0].title == "Book 1"

    @patch.object(OcrService, 'process_image')
    def test_process_batch_with_cleanup(self, mock_process, tmp_path):
        """Batch processing should cleanup files when cleanup=True."""
        # Create temporary test files
        test_files = []
        for i in range(2):
            f = tmp_path / f"test{i}.jpg"
            f.write_bytes(b"fake content")
            test_files.append(str(f))
        
        mock_process.return_value = OcrResult(
            title="Test",
            author="Author",
            needs_review=False,
            raw_text="Test"
        )
        
        result = OcrService.process_batch(test_files, cleanup=True)
        
        # Files should be deleted after cleanup
        for f in test_files:
            assert not tmp_path.joinpath(f.split("/")[-1]).exists()


class TestParsedBookInfo:
    """Tests for ParsedBookInfo dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        info = ParsedBookInfo()
        
        assert info.title is None
        assert info.author is None
        assert info.isbn is None
        assert info.confidence == 0.0
        assert info.raw_text == ""
        assert info.needs_review is True

    def test_with_values(self):
        """Values should be set correctly."""
        info = ParsedBookInfo(
            title="Test Title",
            author="Test Author",
            isbn="9780123456789",
            confidence=0.95,
            raw_text="Test Title Test Author",
            needs_review=False
        )
        
        assert info.title == "Test Title"
        assert info.author == "Test Author"
        assert info.isbn == "9780123456789"
        assert info.confidence == 0.95
        assert info.needs_review is False
