import logging
from typing import Tuple

# -------------- logger --------------

logger = logging.getLogger("cf-buir")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("[%(name)s] %(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"))
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


# -------------- custom types --------------

Interaction = Tuple[int, int] 
