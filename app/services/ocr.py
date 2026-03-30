import os
import re
import easyocr
from dataclasses import dataclass
from typing import List, Optional

from app.schemas.book import OcrResult, BatchScanResponse


@dataclass
class ParsedBookInfo:
    title: Optional[str]
    author: Optional[str]
    isbn: Optional[str]
    confidence: float
    raw_text: str
    needs_review: bool


class OcrService:
    _reader: Optional[easyocr.Reader] = None

    @classmethod
    def get_reader(cls) -> easyocr.Reader:
        if cls._reader is None:
            cls._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return cls._reader

    @classmethod
    def process_image(cls, image_path: str) -> OcrResult:
        reader = cls.get_reader()
        results = reader.readtext(image_path)

        if not results:
            return OcrResult(
                raw_text="",
                needs_review=True,
                image_path=image_path
            )

        raw_texts = []
        text_blocks = []

        for (bbox, text, confidence) in results:
            if text.strip():
                raw_texts.append(text.strip())
                text_blocks.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': bbox
                })

        raw_text = ' '.join(raw_texts)
        avg_confidence = sum(b['confidence'] for b in text_blocks) / len(text_blocks) if text_blocks else 0

        parsed = cls._parse_text_blocks(text_blocks)

        needs_review = parsed.title is None or parsed.author is None

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
    def _parse_text_blocks(cls, blocks: List[dict]) -> ParsedBookInfo:
        texts = [b['text'] for b in blocks]
        full_text = ' '.join(texts)

        isbn_match = cls._extract_isbn(full_text)
        title, author = cls._extract_title_author(texts)

        return ParsedBookInfo(
            title=title,
            author=author,
            isbn=isbn_match,
            confidence=0.0,
            raw_text=full_text,
            needs_review=title is None or author is None
        )

    @classmethod
    def _extract_isbn(cls, text: str) -> Optional[str]:
        patterns = [
            r'ISBN[:\s]*([0-9X-]{10,17})',
            r'ISBN\s*([0-9X-]{10,17})',
            r'([0-9]{13})',
            r'([0-9]{10})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace('-', '')

        return None

    @classmethod
    def _extract_title_author(cls, texts: List[str]) -> tuple[Optional[str], Optional[str]]:
        if not texts:
            return None, None

        author_indicators = [
            'by', 'written by', 'author:', 'author', '著',
            '-', '—', '–'
        ]

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
            if heuristics['title']:
                title = heuristics['title']
                author = heuristics['author']

        title = cls._clean_text(title) if title else None
        author = cls._clean_text(author) if author else None

        return title, author

    @classmethod
    def _apply_title_heuristics(cls, texts: List[str]) -> dict:
        if len(texts) == 1:
            return {'title': texts[0], 'author': None}

        if len(texts) == 2:
            if len(texts[0]) > len(texts[1]):
                return {'title': texts[0], 'author': texts[1]}
            else:
                return {'title': texts[1], 'author': texts[0]}

        first_is_long = len(texts[0]) > 30
        second_is_short = len(texts[1]) < 20

        if first_is_long and second_is_short:
            return {'title': texts[0], 'author': texts[1]}
        elif not first_is_long:
            possible_title = ' '.join(texts[:2])
            possible_author = texts[2] if len(texts) > 2 else None
            return {'title': possible_title, 'author': possible_author}

        title_candidates = []
        author_candidates = []

        for text in texts:
            if len(text) > 40:
                title_candidates.append((len(text), text))
            elif 3 < len(text) < 25:
                author_candidates.append((len(text), text))

        title_candidates.sort(reverse=True)
        author_candidates.sort(reverse=True)

        title = title_candidates[0][1] if title_candidates else None
        author = author_candidates[0][1] if author_candidates else None

        return {'title': title, 'author': author}

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = re.sub(r'[^\w\s\-.,;:!?\'"()]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def process_batch(cls, image_paths: List[str]) -> BatchScanResponse:
        successful = []
        needs_review = []

        for path in image_paths:
            result = cls.process_image(path)
            if result.needs_review:
                needs_review.append(result)
            else:
                successful.append(result)

        return BatchScanResponse(
            successful=successful,
            needs_review=needs_review,
            total_processed=len(image_paths)
        )
