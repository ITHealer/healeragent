import logging
import os
from pathlib import Path

PROJECT_ROOT_WORKDIR = Path(os.getcwd())
LOG_DIR_PATH = PROJECT_ROOT_WORKDIR / "logs" 
LOG_FILE_NAME = "app.log"

os.makedirs(LOG_DIR_PATH, exist_ok=True)

LOG_FILE_FULL_PATH = LOG_DIR_PATH / LOG_FILE_NAME

_configured_loggers = {}

def setup_logger(logger_name: str = 'app_logger',
                 log_level: int = logging.INFO,
                 force_reconfigure: bool = False) -> logging.Logger:
    """
    Thiết lập và trả về một instance logger.

    Args:
        logger_name (str): Tên của logger. Thường dùng __name__ để lấy tên module.
        log_level (int): Cấp độ log (ví dụ: logging.INFO, logging.DEBUG).
        force_reconfigure (bool): Nếu True, buộc cấu hình lại logger ngay cả khi nó đã tồn tại.

    Returns:
        logging.Logger: Instance logger đã được cấu hình.
    """
    if logger_name in _configured_loggers and not force_reconfigure:
        return _configured_loggers[logger_name]

    logger = logging.getLogger(logger_name)

    if force_reconfigure or not logger.handlers:
        logger.handlers = []

    logger.setLevel(log_level) 
    logger.propagate = False

    try:
        file_handler = logging.FileHandler(LOG_FILE_FULL_PATH, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi thiết lập file handler cho logger '{logger_name}' tới file '{LOG_FILE_FULL_PATH}': {e}")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) 
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    _configured_loggers[logger_name] = logger
    return logger