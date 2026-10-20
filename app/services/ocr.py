"""
OCR service for extracting text from book spine images.

This module provides OCR functionality using EasyOCR to:
- Extract text from uploaded book spine images
- Parse title, author, and ISBN from the extracted text
- Handle errors gracefully and flag results needing review
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image

import easyocr

from app.schemas.book import OcrResult, BatchScanResponse
from app.services.upload import cleanup_files, Path


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = ['en', 'pt']


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

    # Book-spine photos often arrive sideways (text runs along the spine).
    # OCR the base image first; if too little readable text is found, retry
    # at 90/180/270 degrees and keep the orientation with the most text.
    AUTO_ROTATE: bool = os.environ.get("OCR_AUTO_ROTATE", "1") not in {"0", "false", "False"}
    ROTATION_TEXT_THRESHOLD: float = 5.0

    # Words that signal the following/adjacent text is the author.
    AUTHOR_INDICATOR_WORDS = {
        "by", "written by", "author", "por", "escrito por", "autor", "autoria",
    }

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
    def _preprocess_image(cls, image_path: str) -> str:
        """Resize large images and gently enhance for better OCR."""
        MAX_DIM = 1920
        img = Image.open(image_path).convert("L")
        w, h = img.size
        if max(w, h) > MAX_DIM:
            ratio = MAX_DIM / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.5)
        resized_path = image_path.rsplit(".", 1)
        out_path = f"{resized_path[0]}_processed.{resized_path[1]}"
        img.save(out_path)
        logger.info(f"Preprocessed {image_path} ({w}x{h}) -> {out_path}")
        return out_path

    @classmethod
    def process_image(cls, image_path: str) -> OcrResult:
        """
        Process a single image and extract book information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OcrResult with extracted data and metadata
        """
        processed_path = cls._preprocess_image(image_path)

        try:
            results, angle = cls._readtext_with_orientation(processed_path)
            if angle:
                logger.info(f"Auto-rotated image by {angle} degrees for OCR")
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
        
        title_ok = parsed.title and len(parsed.title) >= 2 and re.search(r'[A-Za-zÀ-ÿ]', parsed.title)
        author_ok = parsed.author and len(parsed.author) >= 2 and re.search(r'[A-Za-zÀ-ÿ]', parsed.author)
        needs_review = (
            not title_ok or
            not author_ok or
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
    def _score_results(cls, results: list) -> float:
        """Score OCR results by amount of readable text weighted by confidence."""
        score = 0.0
        for item in results:
            text = item[1] if len(item) > 1 else ""
            confidence = item[2] if len(item) > 2 else 0.0
            score += sum(c.isalpha() for c in str(text)) * float(confidence)
        return score

    @classmethod
    def _readtext_with_orientation(cls, image_path: str) -> tuple[list, int]:
        """
        Run OCR, auto-detecting image orientation.

        Tries the original orientation first. If too little readable text is
        found (common for sideways book-spine photos), retries at 90/180/270
        degrees and returns the results with the highest text score.

        Returns:
            Tuple of (results, angle) where angle is 0 for the original.
        """
        reader = cls.get_reader()
        img = Image.open(image_path).convert("RGB")
        results = reader.readtext(np.array(img))

        if not cls.AUTO_ROTATE:
            return results, 0

        best_score = cls._score_results(results)
        best = (best_score, 0, results)

        if best_score < cls.ROTATION_TEXT_THRESHOLD:
            for angle in (90, 180, 270):
                try:
                    rotated = np.array(img.rotate(angle, expand=True))
                    rotated_results = reader.readtext(rotated)
                except Exception as e:
                    logger.warning(f"Rotation {angle} OCR failed: {e}")
                    continue
                score = cls._score_results(rotated_results)
                if score > best[0]:
                    best = (score, angle, rotated_results)

        return best[2], best[1]

    @staticmethod
    def _bbox_height(bbox) -> Optional[float]:
        """Return the pixel height of a bounding box, or None if unavailable."""
        if not bbox:
            return None
        try:
            ys = [float(point[1]) for point in bbox]
        except (TypeError, IndexError, ValueError):
            return None
        height = max(ys) - min(ys)
        return height if height > 0 else None

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
        title, author = cls._extract_title_author_from_blocks(blocks)
        
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

        Backwards-compatible wrapper that accepts plain strings. Prefer
        `_extract_title_author_from_blocks` when bounding boxes are available.

        Args:
            texts: Ordered list of text blocks from OCR

        Returns:
            Tuple of (title, author)
        """
        blocks = [
            {'text': t, 'bbox': None} if isinstance(t, str) else t
            for t in texts
        ]
        return cls._extract_title_author_from_blocks(blocks)

    @classmethod
    def _extract_title_author_from_blocks(
        cls, blocks: list[dict]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract title and author from OCR blocks.

        Strategy:
        1. Explicit author indicators ("by", "author:", "por", ...).
        2. Blocks that look like person names.
        3. Geometry: on a book spine the title uses a larger font than the
           author, so the smallest text block is the author. Falls back to
           text-length heuristics when bounding boxes are unavailable.

        Args:
            blocks: List of dicts with 'text' and optional 'bbox' keys

        Returns:
            Tuple of (title, author)
        """
        items = []
        for block in blocks:
            text = (block.get('text') if isinstance(block, dict) else block) or ""
            text = text.strip()
            if not text:
                continue
            items.append({
                'text': text,
                'bbox': block.get('bbox') if isinstance(block, dict) else None,
            })

        if not items:
            return None, None

        indicator_title, indicator_author = cls._split_by_author_indicator(items)
        if indicator_author:
            title = cls._clean_text(indicator_title) if indicator_title else None
            return title, cls._clean_text(indicator_author)

        name_items = [it for it in items if cls._looks_like_name(it['text'])]
        if name_items:
            author_item = cls._pick_author_item(name_items)
            title = ' '.join(it['text'] for it in items if it is not author_item) or None
            return cls._clean_text(title) if title else None, cls._clean_text(author_item['text'])

        heuristics = cls._apply_title_heuristics(items)
        title = heuristics.get('title')
        author = heuristics.get('author')
        return (
            cls._clean_text(title) if title else None,
            cls._clean_text(author) if author else None,
        )

    @classmethod
    def _split_by_author_indicator(
        cls, items: list[dict]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Detect author indicators such as a standalone "by" block or an inline
        "by John Doe" / "Author: John Doe".

        Returns:
            Tuple of (title, author); (None, None) when no indicator is found.
        """
        inline_pattern = re.compile(
            r'(?:^|[-–—:;,])\s*(written by|by|author|por|autor|autoria)\b\s*:?\s*',
            re.IGNORECASE,
        )

        for i, item in enumerate(items):
            text = item['text']
            normalized = re.sub(
                r'\s+', ' ', text.strip().strip('.:;,-–—').lower()
            )

            if normalized in cls.AUTHOR_INDICATOR_WORDS and i + 1 < len(items):
                title = ' '.join(x['text'] for x in items[:i]) or None
                return title, items[i + 1]['text']

            match = inline_pattern.search(text)
            if match:
                prefix = text[:match.start()].strip(' :;,-–—')
                remainder = text[match.end():].strip(' :;,-–—')
                if remainder and len(remainder) > 1:
                    title_parts = [x['text'] for x in items[:i]]
                    if prefix:
                        title_parts.append(prefix)
                    title = ' '.join(title_parts) or None
                    return title, remainder

        return None, None

    @classmethod
    def _looks_like_name(cls, text: str) -> bool:
        """
        Check if text looks like an author name.

        Handles "Agatha Christie", "James A. Michener", "F. Scott Fitzgerald"
        and the CamelCase form "AgathaChristie".
        """
        text = text.strip().strip('.,;:')
        if not text:
            return False

        def is_word(part: str) -> bool:
            return len(part) >= 2 and part[0].isupper() and any(c.isalpha() for c in part)

        def is_initial(part: str) -> bool:
            core = part.strip('.')
            return len(core) == 1 and core.isalpha() and core.isupper()

        parts = text.split()

        if len(parts) == 2:
            return is_word(parts[0]) and is_word(parts[1])

        if 2 <= len(parts) <= 4:
            if all(is_word(p) or is_initial(p) for p in parts) and any(is_initial(p) for p in parts):
                return True

        if len(parts) == 1 and len(text) >= 5:
            uc_positions = [i for i, c in enumerate(text) if c.isupper()]
            if len(uc_positions) >= 2:
                splits = []
                for i, pos in enumerate(uc_positions):
                    end = uc_positions[i + 1] if i + 1 < len(uc_positions) else len(text)
                    splits.append(text[pos:end])
                if len(splits) == 2 and all(len(s) >= 2 for s in splits):
                    return True

        return False

    @classmethod
    def _pick_author_item(cls, name_items: list[dict]) -> dict:
        """Choose the most likely author among name-like blocks."""
        if len(name_items) == 1:
            return name_items[0]
        heights = [cls._bbox_height(it['bbox']) for it in name_items]
        if all(heights):
            return min(name_items, key=lambda it: cls._bbox_height(it['bbox']))
        return name_items[-1]

    @classmethod
    def _apply_title_heuristics(cls, items: list[dict]) -> dict:
        """
        Apply heuristics to guess title and author when no author indicator or
        name-like block is found.

        Uses bounding-box height when available (smallest font = author) and
        falls back to text length otherwise.

        Args:
            items: List of dicts with 'text' and optional 'bbox' keys

        Returns:
            Dict with 'title' and 'author' keys
        """
        if len(items) == 1:
            return {'title': items[0]['text'], 'author': None}

        heights = [cls._bbox_height(it['bbox']) for it in items]
        has_geometry = all(heights)

        if len(items) == 2:
            if has_geometry:
                h0, h1 = heights
                largest = max(h0, h1)
                if largest > 0 and abs(h0 - h1) / largest > 0.15:
                    author_idx = 0 if h0 < h1 else 1
                    title_idx = 1 - author_idx
                    return {
                        'title': items[title_idx]['text'],
                        'author': items[author_idx]['text'],
                    }
            if len(items[0]['text']) > len(items[1]['text']):
                return {'title': items[0]['text'], 'author': items[1]['text']}
            return {'title': items[1]['text'], 'author': items[0]['text']}

        alpha_items = [it for it in items if re.search(r'[A-Za-zÀ-ÿ]', it['text'])]
        if has_geometry and alpha_items:
            author_item = min(alpha_items, key=lambda it: cls._bbox_height(it['bbox']))
        else:
            author_item = min(items, key=lambda it: len(it['text']))

        title = ' '.join(it['text'] for it in items if it is not author_item) or None
        return {'title': title, 'author': author_item['text']}

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
