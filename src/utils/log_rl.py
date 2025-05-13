import logging


def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that writes to the specified file.
    """
    logger = logging.getLogger('MRI_RL')
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        # Console handler (optional)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        # Formatter
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger