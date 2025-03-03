import logging
import threading
import os

class Logger:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, logger_name: str):
        with cls._lock:
            if logger_name not in cls._instances:
                instance = super(Logger, cls).__new__(cls)
                instance._initialize(logger_name)
                cls._instances[logger_name] = instance
        return cls._instances[logger_name]

    def _initialize(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Create log directory
        log_dir = ".logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{logger_name}.log")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_level(self, level: int):
        self.logger.setLevel(level)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)