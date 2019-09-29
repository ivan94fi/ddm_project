import logging
from pprint import pprint
import tsfresh

levels = """
CRITICAL 50
ERROR    40
WARNING  30
INFO     20
DEBUG    10
NOTSET    0
------------
"""


def _get_loggers():
    return [logging.getLogger(name) for name in logging.root.manager.loggerDict]


print(levels)
logger = logging.getLogger("my_logger")
print(logger.getEffectiveLevel())
logger.setLevel(logging.CRITICAL)
print(logger.getEffectiveLevel())

print("=" * 40)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

pprint(_get_loggers())
