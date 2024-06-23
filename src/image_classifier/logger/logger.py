import logging
import logging.config
from pathlib import Path

from image_classifier.utils.file_loaders import load_yaml


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    def __init__(
        self, cyan: str, black: str, yellow: str, red: str, bold_red: str, reset: str, format: str, datefmt: str
    ) -> None:
        self.datefmt = datefmt
        self._formats = {
            logging.DEBUG: cyan + format + reset,
            logging.INFO: black + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        """Checking the log level and applying the corresponding color format by creating a new Formatter"""

        log_level_fmt = self._formats.get(record.levelno)
        new_formatter = logging.Formatter(fmt=log_level_fmt, datefmt=self.datefmt)

        return new_formatter.format(record=record)


class StdoutFilter(logging.Filter):
    """Filter to only log messages with level INFO or lower"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


class StderrFilter(logging.Filter):
    """Filter to only log messages with level WARNING or higher"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


def initialize_root_logger(config: dict | Path = Path(__file__).parent.joinpath("config.yaml")) -> None:
    if isinstance(config, Path):
        config = load_yaml(file_path=config)

    logging.config.dictConfig(config=config)
