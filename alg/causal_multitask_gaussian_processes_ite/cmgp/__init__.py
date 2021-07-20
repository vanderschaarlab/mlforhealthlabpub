import sys

from . import logger  # noqa: F401
from .cmgp import CMGP  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
