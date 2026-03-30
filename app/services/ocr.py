"""
OCR service for extracting text from book spine images.

This module provides OCR functionality using EasyOCR to:
- Extract text from uploaded book spine images
- Parse title, author, and ISBN from the extracted text
- Handle errors gracefully and flag results needing review
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import easyocr

from app.schemas.book import OcrResult, BatchScanResponse
from app.services.upload import cleanup_files, Path


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = ['en']


@dataclass
class ParsedBookInfo:
    """Container for parsed book information from OCR."""
    title: Optional[str] = None
    author: Optional[str] = None
    isbn: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""
    needs_review: bool = True


class OcrService:
    """
    Service for performing OCR on book spine images.
    
    Uses EasyOCR for text recognition with heuristics for
    extracting structured book information (title, author, ISBN).
    """
    
    _reader: Optional[easyocr.Reader] = None
    
    @classmethod
    def initialize(cls, languages: list[str] = None, gpu: bool = False) -> None:
        """
        Initialize the OCR reader with specified languages.
        
        Should be called once at application startup.
        
        Args:
            languages: List of language codes (e.g., ['en', 'pt'])
            gpu: Whether to use GPU acceleration
        """
        if cls._reader is None:
            langs = languages or SUPPORTED_LANGUAGES
            logger.info(f"Initializing EasyOCR reader with languages: {langs}")
            cls._reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    
    @classmethod
    def get_reader(cls) -> easyocr.Reader:
        """Get or create the OCR reader instance."""
        if cls._reader is None:
            cls.initialize()
        return cls._reader
    
    @classmethod
    def process_image(cls, image_path: str) -> OcrResult:
        """
        Process a single image and extract book information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OcrResult with extracted data and metadata
        """
        reader = cls.get_reader()
        
        try:
            results = reader.readtext(image_path)
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return OcrResult(
                raw_text="",
                needs_review=True,
                image_path=image_path
            )
        
        if not results:
            return OcrResult(
                raw_text="",
                needs_review=True,
                image_path=image_path
            )
        
        text_blocks = []
        raw_texts = []
        
        for (bbox, text, confidence) in results:
            if text.strip():
                raw_texts.append(text.strip())
                text_blocks.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        raw_text = ' '.join(raw_texts)
        avg_confidence = (
            sum(b['confidence'] for b in text_blocks) / len(text_blocks)
            if text_blocks else 0.0
        )
        
        parsed = cls._parse_text_blocks(text_blocks)
        parsed.confidence = avg_confidence
        
        needs_review = (
            parsed.title is None or 
            parsed.author is None or 
            avg_confidence < 0.5
        )
        
        return OcrResult(
            title=parsed.title,
            author=parsed.author,
            isbn=parsed.isbn,
            confidence=avg_confidence,
            raw_text=raw_text,
            needs_review=needs_review,
            image_path=image_path
        )
    
    @classmethod
    def _parse_text_blocks(cls, blocks: list[dict]) -> ParsedBookInfo:
        """
        Parse OCR text blocks to extract book information.
        
        Args:
            blocks: List of dicts with 'text', 'confidence', 'bbox' keys
            
        Returns:
            ParsedBookInfo with extracted title, author, ISBN
        """
        texts = [b['text'] for b in blocks]
        full_text = ' '.join(texts)
        
        isbn = cls._extract_isbn(full_text)
        title, author = cls._extract_title_author(texts)
        
        return ParsedBookInfo(
            title=title,
            author=author,
            isbn=isbn,
            confidence=0.0,
            raw_text=full_text,
            needs_review=title is None or author is None
        )
    
    @classmethod
    def _extract_isbn(cls, text: str) -> Optional[str]:
        """
        Extract ISBN from text using regex patterns.
        
        Args:
            text: Combined OCR text
            
        Returns:
            ISBN string or None if not found
        """
        patterns = [
            r'ISBN[:\s]*([0-9X-]{10,17})',
            r'ISBN\s*([0-9X-]{10,17})',
            r'([0-9]{13})(?:\s|$)',
            r'([0-9]{10})(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                isbn = match.group(1).replace('-', '')
                if len(isbn) >= 10:
                    return isbn
        
        return None
    
    @classmethod
    def _extract_title_author(cls, texts: list[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Extract title and author from ordered text blocks.
        
        Looks for author indicators first, then falls back to
        heuristics based on text length and position.
        
        Args:
            texts: Ordered list of text blocks from OCR
            
        Returns:
            Tuple of (title, author)
        """
        if not texts:
            return None, None
        
        author_indicators = ['by', 'written by', 'author:', 'author', '著', '-', '—', '–']
        
        title = None
        author = None
        title_end_idx = 0
        
        for i, text in enumerate(texts):
            upper_text = text.upper()
            
            for indicator in author_indicators:
                if indicator.upper() in upper_text:
                    candidate_author = text.split(indicator, 1)[-1].strip()
                    if candidate_author and len(candidate_author) > 1:
                        author = candidate_author
                        title_end_idx = i
                        break
            
            if author:
                break
        
        if author:
            title = ' '.join(texts[:title_end_idx]) if title_end_idx > 0 else texts[0]
        else:
            heuristics = cls._apply_title_heuristics(texts)
            title = heuristics.get('title')
            author = heuristics.get('author')
        
        title = cls._clean_text(title) if title else None
        author = cls._clean_text(author) if author else None
        
        return title, author
    
    @classmethod
    def _apply_title_heuristics(cls, texts: list[str]) -> dict:
        """
        Apply heuristics to guess title and author from text blocks.
        
        Strategy:
        - Single text: treat as title only
        - Two texts: longer is title, shorter is author
        - Multiple texts: use length-based heuristics
        
        Args:
            texts: List of text blocks
            
        Returns:
            Dict with 'title' and 'author' keys
        """
        if len(texts) == 1:
            return {'title': texts[0], 'author': None}
        
        if len(texts) == 2:
            if len(texts[0]) > len(texts[1]):
                return {'title': texts[0], 'author': texts[1]}
            return {'title': texts[1], 'author': texts[0]}
        
        first_is_long = len(texts[0]) > 30
        second_is_short = len(texts[1]) < 20
        
        if first_is_long and second_is_short:
            return {'title': texts[0], 'author': texts[1]}
        elif not first_is_long:
            return {
                'title': ' '.join(texts[:2]),
                'author': texts[2] if len(texts) > 2 else None
            }
        
        title_candidates = [(len(t), t) for t in texts if len(t) > 40]
        author_candidates = [(len(t), t) for t in texts if 3 < len(t) < 25]
        
        title_candidates.sort(reverse=True)
        author_candidates.sort(reverse=True)
        
        return {
            'title': title_candidates[0][1] if title_candidates else None,
            'author': author_candidates[0][1] if author_candidates else None
        }
    
    @classmethod
    def _clean_text(cls, text: str) -> str:
        """
        Clean extracted text by removing unwanted characters.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        text = re.sub(r'[^\w\s\-.,;:!?\'"()]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @classmethod
    def process_batch(
        cls,
        image_paths: list[str] | list[Path],
        cleanup: bool = True
    ) -> BatchScanResponse:
        """
        Process multiple images and categorize results.
        
        Args:
            image_paths: List of image file paths (str or Path)
            cleanup: Whether to delete processed images after OCR
            
        Returns:
            BatchScanResponse with successful and needs_review results
        """
        successful = []
        needs_review = []
        paths_to_cleanup: list[Path] = []
        
        for path in image_paths:
            path_str = str(path)
            paths_to_cleanup.append(Path(path_str))
            result = cls.process_image(path_str)
            
            if result.needs_review:
                needs_review.append(result)
            else:
                successful.append(result)
        
        if cleanup:
            cleanup_files(paths_to_cleanup)
        
        return BatchScanResponse(
            successful=successful,
            needs_review=needs_review,
            total_processed=len(image_paths)
        )
