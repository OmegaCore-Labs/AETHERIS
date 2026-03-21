"""
Security Utilities

Model encryption, integrity verification, and output sanitization.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
import base64


class SecurityManager:
    """
    Security utilities for AETHERIS.

    Features:
    - Model encryption/decryption
    - Integrity verification (SHA-256)
    - Output sanitization
    - API key management
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize security manager.

        Args:
            key: Optional encryption key (generates new if not provided)
        """
        if key:
            self._cipher = Fernet(key)
        else:
            self._cipher = None

    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()

    def encrypt_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt a model directory.

        Args:
            model_path: Path to model directory
            output_path: Output path for encrypted file

        Returns:
            Path to encrypted file
        """
        if not self._cipher:
            raise ValueError("Encryption key not set. Provide key in constructor.")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Collect all files
        files_data = {}
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    files_data[str(file_path.relative_to(model_path))] = f.read()

        # Serialize and encrypt
        data = json.dumps(files_data, default=lambda x: base64.b64encode(x).decode())
        encrypted = self._cipher.encrypt(data.encode())

        output = Path(output_path) if output_path else model_path.with_suffix('.enc')
        with open(output, 'wb') as f:
            f.write(encrypted)

        return str(output)

    def decrypt_model(self, encrypted_path: str, output_dir: str) -> str:
        """
        Decrypt an encrypted model.

        Args:
            encrypted_path: Path to encrypted file
            output_dir: Output directory for decrypted model

        Returns:
            Path to decrypted model directory
        """
        if not self._cipher:
            raise ValueError("Encryption key not set. Provide key in constructor.")

        encrypted_path = Path(encrypted_path)
        if not encrypted_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")

        # Read and decrypt
        with open(encrypted_path, 'rb') as f:
            encrypted = f.read()

        decrypted = self._cipher.decrypt(encrypted)
        data = json.loads(decrypted.decode())

        # Write files
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in data.items():
            full_path = output_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(base64.b64decode(content))

        return str(output_dir)

    def verify_integrity(self, file_path: str, expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify file integrity using SHA-256.

        Args:
            file_path: Path to file
            expected_hash: Optional expected hash for comparison

        Returns:
            Dictionary with hash and verification result
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": "File not found", "verified": False}

        # Compute hash
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)

        computed_hash = sha256.hexdigest()

        result = {
            "file": str(file_path),
            "hash": computed_hash,
            "size_bytes": file_path.stat().st_size,
            "verified": False
        }

        if expected_hash:
            result["verified"] = computed_hash == expected_hash
            result["expected_hash"] = expected_hash

        return result

    def sanitize_output(self, text: str, max_length: int = 10000) -> str:
        """
        Sanitize model output.

        Args:
            text: Output text to sanitize
            max_length: Maximum length

        Returns:
            Sanitized text
        """
        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Remove control characters
        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        return text

    def mask_api_key(self, key: str, visible: int = 4) -> str:
        """
        Mask API key for display.

        Args:
            key: API key
            visible: Number of characters to show at start and end

        Returns:
            Masked key
        """
        if len(key) <= visible * 2:
            return "*" * len(key)

        start = key[:visible]
        end = key[-visible:]
        return f"{start}{'*' * (len(key) - visible * 2)}{end}"

    def validate_api_key(self, key: str, min_length: int = 20) -> bool:
        """
        Validate API key format.

        Args:
            key: API key to validate
            min_length: Minimum length

        Returns:
            True if key appears valid
        """
        if not key or len(key) < min_length:
            return False

        # Check for common patterns
        if key.startswith('sk-') or key.startswith('hf_'):
            return True

        # Hex-like keys
        import re
        if re.match(r'^[a-fA-F0-9]{32,}$', key):
            return True

        return True

    def secure_hash(self, data: str) -> str:
        """
        Create a secure hash of data.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash
        """
        return hashlib.sha256(data.encode()).hexdigest()
