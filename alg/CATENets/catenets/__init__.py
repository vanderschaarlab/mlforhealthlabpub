import sys

from . import logger  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
