# utils/logger.py

import logging

def get_logger(name: str = "mlflowpy") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console = logging.StreamHandler()
        console.setFormatter(formatter)

        logger.addHandler(console)
        logger.propagate = False

    return logger
