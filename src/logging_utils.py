from __future__ import annotations

import logging


def setup_logger(name: str, rank: int = 0, world_size: int = 1) -> logging.Logger:
    logger_name = f"{name}.rank{rank}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | [rank {rank}/{world_size}] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
