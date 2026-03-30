"""
Tests for the upload service module.

Covers:
- Filename sanitization
- File size validation
- Extension validation
- Save upload functions
- Cleanup functions
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from app.services.upload import (
    sanitize_filename,
    validate_file_size,
    validate_extension,
    UploadError,
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
)
from app.services.upload import save_upload, save_multiple_uploads, cleanup_files


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_normal_filename_unchanged(self):
        """Normal filenames should pass through unchanged."""
        assert sanitize_filename("book.jpg") == "book.jpg"
        assert sanitize_filename("my_book-cover.png") == "my_book-cover.png"

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""
        assert sanitize_filename("../etc/passwd") == "passwd"
        assert sanitize_filename("../../../root/.bashrc") == "bashrc"
        assert sanitize_filename("/absolute/path/file.txt") == "file.txt"

    def test_windows_path_traversal_blocked(self):
        """Windows-style path traversal should be blocked."""
        assert sanitize_filename("C:\\Windows\\System32\\config.sys") == "config.sys"
        assert sanitize_filename("..\\..\\windows\\file.dll") == "file.dll"

    def test_dangerous_characters_removed(self):
        """Dangerous characters should be replaced with underscore."""
        assert sanitize_filename("file<script>alert(1)</script>.jpg") == "file_script_alert____script_.jpg"
        assert sanitize_filename("book$%^&*.png") == "book_____.png"

    def test_hidden_files_get_random_name(self):
        """Hidden files (starting with dot) should get random names."""
        result = sanitize_filename(".htaccess")
        assert not result.startswith(".")
        assert result.startswith("file_")

    def test_empty_filename_gets_random_name(self):
        """Empty filenames should get random names."""
        result = sanitize_filename("")
        assert result.startswith("file_")

    def test_only_extension_gets_random_name(self):
        """Filenames with just extension should get random names."""
        result = sanitize_filename(".jpg")
        assert result.startswith("file_")


class TestValidateFileSize:
    """Tests for file size validation."""

    def test_small_file_passes(self):
        """Files under limit should pass."""
        content = b"small content"
        assert validate_file_size(content) is True

    def test_exactly_at_limit_passes(self):
        """Files exactly at limit should pass."""
        content = b"x" * MAX_FILE_SIZE_BYTES
        assert validate_file_size(content) is True

    def test_over_limit_raises_error(self):
        """Files over limit should raise UploadError."""
        content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(UploadError) as exc_info:
            validate_file_size(content)
        assert str(MAX_FILE_SIZE_BYTES // (1024 * 1024)) in str(exc_info.value)


class TestValidateExtension:
    """Tests for file extension validation."""

    @pytest.mark.parametrize("ext", ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
    def test_allowed_extensions_pass(self, ext):
        """All allowed extensions should pass."""
        assert validate_extension(f"file{ext}") is True

    def test_uppercase_extension_passes(self):
        """Uppercase extensions should be accepted."""
        assert validate_extension("file.JPG") is True
        assert validate_extension("file.PNG") is True

    def test_no_extension_fails(self):
        """Files without extension should fail."""
        with pytest.raises(UploadError) as exc_info:
            validate_extension("file")
        assert "not allowed" in str(exc_info.value)

    def test_disallowed_extension_fails(self):
        """Disallowed extensions should fail."""
        with pytest.raises(UploadError) as exc_info:
            validate_extension("file.exe")
        assert "not allowed" in str(exc_info.value)
        assert ".exe" in str(exc_info.value)

    def test_multiple_dots_in_filename(self):
        """Extensions with multiple dots should work."""
        assert validate_extension("file.tar.gz") is True


class TestCleanupFiles:
    """Tests for file cleanup functionality."""

    def test_cleanup_existing_file(self, tmp_path):
        """Existing files should be deleted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        assert test_file.exists()
        cleanup_files([test_file])
        assert not test_file.exists()

    def test_cleanup_nonexistent_file(self, tmp_path):
        """Non-existent files should not raise errors."""
        test_file = tmp_path / "nonexistent.txt"
        
        # Should not raise
        cleanup_files([test_file])

    def test_cleanup_multiple_files(self, tmp_path):
        """Multiple files should all be deleted."""
        files = []
        for i in range(3):
            f = tmp_path / f"test{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)
        
        cleanup_files(files)
        
        for f in files:
            assert not f.exists()

    def test_cleanup_empty_list(self):
        """Empty list should not raise errors."""
        cleanup_files([])


class TestSaveUpload:
    """Tests for save_upload function."""

    @pytest.mark.asyncio
    async def test_save_valid_file(self, tmp_path):
        """Valid files should be saved successfully."""
        mock_file = MagicMock()
        mock_file.filename = "book.jpg"
        mock_file.read = AsyncMock(return_value=b"fake image content")
        
        filepath = await save_upload(mock_file, upload_dir=tmp_path)
        
        assert filepath.exists()
        assert filepath.read_bytes() == b"fake image content"
        assert filepath.parent == tmp_path

    @pytest.mark.asyncio
    async def test_save_without_filename_raises(self, tmp_path):
        """Files without filename should raise UploadError."""
        mock_file = MagicMock()
        mock_file.filename = None
        
        with pytest.raises(UploadError) as exc_info:
            await save_upload(mock_file, upload_dir=tmp_path)
        assert "No filename" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_invalid_extension_raises(self, tmp_path):
        """Invalid extension should raise UploadError."""
        mock_file = MagicMock()
        mock_file.filename = "virus.exe"
        mock_file.read = AsyncMock(return_value=b"content")
        
        with pytest.raises(UploadError) as exc_info:
            await save_upload(mock_file, upload_dir=tmp_path)
        assert "not allowed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_oversized_file_raises(self, tmp_path):
        """Oversized files should raise UploadError."""
        mock_file = MagicMock()
        mock_file.filename = "large.jpg"
        mock_file.read = AsyncMock(return_value=b"x" * (MAX_FILE_SIZE_BYTES + 1))
        
        with pytest.raises(UploadError) as exc_info:
            await save_upload(mock_file, upload_dir=tmp_path)
        assert "exceeds" in str(exc_info.value)


class TestSaveMultipleUploads:
    """Tests for save_multiple_uploads function."""

    @pytest.mark.asyncio
    async def test_save_multiple_valid_files(self, tmp_path):
        """Multiple valid files should all be saved."""
        files = []
        for i in range(3):
            mock_file = MagicMock()
            mock_file.filename = f"book{i}.jpg"
            mock_file.read = AsyncMock(return_value=f"content{i}".encode())
            files.append(mock_file)
        
        paths = await save_multiple_uploads(files, upload_dir=tmp_path)
        
        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    @pytest.mark.asyncio
    async def test_save_mixed_valid_invalid(self, tmp_path):
        """Mixed valid/invalid files should save valid ones."""
        valid_file = MagicMock()
        valid_file.filename = "valid.jpg"
        valid_file.read = AsyncMock(return_value=b"valid")
        
        invalid_file = MagicMock()
        invalid_file.filename = "invalid.exe"
        invalid_file.read = AsyncMock(return_value=b"invalid")
        
        paths = await save_multiple_uploads([valid_file, invalid_file], upload_dir=tmp_path)
        
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].read_bytes() == b"valid"

    @pytest.mark.asyncio
    async def test_save_all_invalid_raises(self, tmp_path):
        """All invalid files should raise UploadError."""
        files = []
        for i in range(3):
            mock_file = MagicMock()
            mock_file.filename = f"file{i}.exe"
            mock_file.read = AsyncMock(return_value=b"content")
            files.append(mock_file)
        
        with pytest.raises(UploadError) as exc_info:
            await save_multiple_uploads(files, upload_dir=tmp_path)
        assert "Failed to save any files" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_empty_list(self, tmp_path):
        """Empty list should return empty list."""
        paths = await save_multiple_uploads([], upload_dir=tmp_path)
        assert paths == []
