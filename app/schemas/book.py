from pydantic import BaseModel, ConfigDict, field_validator


class BookBase(BaseModel):
    title: str
    author: str
    isbn: str | None = None
    read: bool = False
    rating: int | None = None
    notes: str | None = None
    needs_review: bool = False

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v


class BookCreate(BookBase):
    pass


class BookUpdate(BaseModel):
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    read: bool | None = None
    rating: int | None = None
    notes: str | None = None
    needs_review: bool | None = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v


class Book(BookBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


class OcrResult(BaseModel):
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    confidence: float = 0.0
    raw_text: str = ""
    needs_review: bool = True
    image_path: str | None = None


class BatchScanResponse(BaseModel):
    successful: list[OcrResult]
    needs_review: list[OcrResult]
    total_processed: int
