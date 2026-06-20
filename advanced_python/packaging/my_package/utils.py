"""
my_package/utils.py

Utility functions for the package.
"""

import logging

def format_number(number: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{number:.{decimals}f}"


def log_message(message: str, level: str = "info"):
    """Log a message with different levels."""
    logger = logging.getLogger(__name__)
    
    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.info(message)