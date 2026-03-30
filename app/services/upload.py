"""
Upload service for handling file uploads securely.

This module provides secure file upload functionality with:
- Path traversal prevention
- File size limits
- Extension validation
- Automatic cleanup of processed files
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile


UPLOAD_DIR = Path("uploads")

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class UploadError(Exception):
    """Custom exception for upload-related errors."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename from upload
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    name = Path(filename).name
    name = os.path.basename(name)
    
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.')
    sanitized = ''.join(c if c in allowed_chars else '_' for c in name)
    
    if not sanitized or sanitized.startswith('.'):
        sanitized = f'file_{uuid.uuid4().hex[:8]}'
    
    return sanitized


def validate_file_size(content: bytes) -> bool:
    """
    Validate that file size is within limits.
    
    Args:
        content: File content bytes
        
    Returns:
        True if file size is acceptable
        
    Raises:
        UploadError: If file exceeds size limit
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise UploadError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
    return True


def validate_extension(filename: str) -> bool:
    """
    Validate that file extension is allowed.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if extension is allowed
        
    Raises:
        UploadError: If extension is not allowed
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"File type {ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return True


async def save_upload(
    upload_file: UploadFile,
    upload_dir: Path = UPLOAD_DIR
) -> Path:
    """
    Save an uploaded file securely.
    
    Args:
        upload_file: FastAPI UploadFile object
        upload_dir: Directory to save files in
        
    Returns:
        Path to saved file
        
    Raises:
        UploadError: If validation fails or save fails
    """
    if not upload_file.filename:
        raise UploadError("No filename provided")
    
    validate_extension(upload_file.filename)
    
    content = await upload_file.read()
    validate_file_size(content)
    
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    ext = Path(upload_file.filename).suffix.lower()
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = upload_dir / safe_filename
    
    with open(filepath, "wb") as f:
        f.write(content)
    
    return filepath


async def save_multiple_uploads(
    upload_files: List[UploadFile],
    upload_dir: Path = UPLOAD_DIR
) -> List[Path]:
    """
    Save multiple uploaded files securely.
    
    Args:
        upload_files: List of FastAPI UploadFile objects
        upload_dir: Directory to save files in
        
    Returns:
        List of paths to saved files
    """
    paths = []
    errors = []
    
    for upload_file in upload_files:
        try:
            filepath = await save_upload(upload_file, upload_dir)
            paths.append(filepath)
        except UploadError as e:
            errors.append(f"{upload_file.filename}: {str(e)}")
    
    if not paths and errors:
        raise UploadError(f"Failed to save any files: {'; '.join(errors)}")
    
    return paths


def cleanup_files(filepaths: List[Path]) -> None:
    """
    Remove uploaded files after processing.
    
    Args:
        filepaths: List of file paths to remove
    """
    for filepath in filepaths:
        try:
            if filepath.exists():
                filepath.unlink()
        except OSError:
            pass
