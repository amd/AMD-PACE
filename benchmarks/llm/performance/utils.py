# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import logging

LOG_DATE_FORMAT = "%d-%b-%y %H:%M:%S"
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelchar)s %(filename)s:%(lineno)d %(name)s: %(message)s"


class SingleLetterLevelFormatter(logging.Formatter):

    def format(self, record):
        # Ensure 'record.levelname' exists and is not empty before accessing index 0
        if record.levelname:
            record.levelchar = record.levelname[0]
        else:
            record.levelchar = "?"  # Fallback if levelname is somehow missing
        return super().format(record)


logger = logging.getLogger("pace-llm-benchmark")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs
fh = logging.FileHandler("pace_llm_benchmark.log")  # Log file handler
ch = logging.StreamHandler()  # Logs also printed to console

# Set formatting
# add formatting for time to be 13-Nov-25 13:14:55.114
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter = SingleLetterLevelFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


PACE_LLM_INFO = logger.info
PACE_LLM_WARNING = logger.warning


def PACE_LLM_ASSERT(condition, message):
    """Custom assert function that logs an error message before raising an AssertionError."""
    if not condition:
        logger.error(f"ASSERTION FAILED: {message}")
        raise AssertionError(message)


def suppress_logging_fn(func):
    """Decorator to suppress logging output from a function."""
    try:
        from pace.utils.logging import suppress_logging_fn

        return suppress_logging_fn(func)
    except ImportError:

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
