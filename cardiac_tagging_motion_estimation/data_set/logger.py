# ----------------------------------------------------------------------------------
# imports
import logging
import os
from enum import Enum
from logging.handlers import RotatingFileHandler

# ----------------------------------------------------------------------------------
#
class LOGGER(Enum):
    STREAM = 0
    FILE = 1
    PRINT = 2
    NONE = 3


LOGGER_TYPE = LOGGER.STREAM
LOGGER_FORMAT = '%(asctime)2s [%(pathname)2s line:%(lineno)d] %(levelname)-6s %(message)s'

# ----------------------------------------------------------------------------------
#  general logging
def get_logger(logger_name='medinfer'):
    logger = logging.getLogger(logger_name)
    if LOGGER_TYPE == LOGGER.STREAM:
        steam_handler = logging.StreamHandler()
        steam_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logger.addHandler(steam_handler)
        logger.setLevel(logging.INFO)
    elif LOGGER_TYPE == LOGGER.FILE:
        rotate_file_handler = RotatingFileHandler(os.path.join(nets.homepath, 'cardiac_pc_recon_nets.log'),
                                                  maxBytes=10240,
                                                  backupCount=10,
                                                  encoding='utf-8')
        rotate_file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logger.addHandler(rotate_file_handler)
        logger.setLevel(logging.INFO)
    return logger