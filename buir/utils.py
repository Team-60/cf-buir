import logging
import torch

# -------------- logger --------------

_logger = logging.getLogger("cf-buir")


def init_logger(filename: str):
    global _logger

    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter("[%(name)s] %(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"))
    _logger.addHandler(handler)

    handlerStdout = logging.StreamHandler()
    handlerStdout.setFormatter(logging.Formatter("[%(name)s] %(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"))
    _logger.addHandler(handlerStdout)

    _logger.setLevel(logging.DEBUG)


def get_logger() -> logging.Logger:
    return _logger


# -------------- device --------------

_device: torch.device


def init_device():
    global _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return _device
