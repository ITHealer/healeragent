import base64
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import json
import logging

from src.utils.config import settings

logger = logging.getLogger(__name__)

class CryptoHelper:
    def __init__(self, key_hex: str):
        """
        Khởi tạo helper với một key dạng chuỗi hex 64 ký tự.
        """
        try:
            self.key = bytes.fromhex(key_hex)
            if len(self.key) != 32:
                raise ValueError("Key dạng hex sau khi chuyển đổi phải có độ dài 32 bytes.")
        except ValueError as e:
            logger.error(f"AES_ENCRYPTION_KEY không hợp lệ. Nó phải là một chuỗi hex gồm 64 ký tự. Lỗi: {e}")
            raise ValueError("AES_ENCRYPTION_KEY phải là một chuỗi hex gồm 64 ký tự.") from e
        except Exception as e:
            logger.error(f"Lỗi không xác định khi khởi tạo CryptoHelper: {e}", exc_info=True)
            raise

    def encrypt(self, data: str) -> Optional[str]:
        """Mã hóa một chuỗi dữ liệu bằng AES GCM."""
        try:
            header = b"header"
            data_bytes = data.encode('utf-8')
            cipher = AES.new(self.key, AES.MODE_GCM)
            cipher.update(header)
            ciphertext, tag = cipher.encrypt_and_digest(data_bytes)
            
            """ Nonce (16 bytes) + Header (6 bytes) + Tag (16 bytes) + Ciphertext"""
            json_payload = {
                'nonce': base64.b64encode(cipher.nonce).decode('utf-8'),
                'header': base64.b64encode(header).decode('utf-8'),
                'tag': base64.b64encode(tag).decode('utf-8'),
                'ciphertext': base64.b64encode(ciphertext).decode('utf-8')
            }
            return json.dumps(json_payload)
        except Exception as e:
            logger.error(f"Lỗi khi mã hóa: {e}", exc_info=True)
            return None

    def decrypt(self, encrypted_json_str: str) -> Optional[str]:
        """Giải mã một chuỗi JSON đã được mã hóa bằng AES GCM."""
        try:
            json_payload = json.loads(encrypted_json_str)
            
            nonce = base64.b64decode(json_payload['nonce'])
            header = base64.b64decode(json_payload['header'])
            tag = base64.b64decode(json_payload['tag'])
            ciphertext = base64.b64decode(json_payload['ciphertext'])

            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            cipher.update(header)
            
            decrypted_data_bytes = cipher.decrypt_and_verify(ciphertext, tag)
            return decrypted_data_bytes.decode('utf-8')
        except (ValueError, KeyError) as e:
            logger.warning(f"Giải mã thất bại. Dữ liệu có thể đã bị thay đổi hoặc key không đúng. Lỗi: {e}")
            return None
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi giải mã: {e}", exc_info=True)
            return None

crypto_helper = CryptoHelper(key_hex=settings.AES_ENCRYPTION_KEY)