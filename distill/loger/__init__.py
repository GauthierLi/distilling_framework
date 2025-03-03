import os 
import sys
import time 
import logging

GREEN = '\033[92m'
RED = '\033[91m'
ORANGE = '\033[93m'

def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message

class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        try:
            level = GlogFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.pathname,
            record.lineno,
            format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)
 
# 获取对象
def get_logger():
    logger = logging.getLogger()
    if os.getenv("DEBUG"):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
 
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(GlogFormatter())
        logger.addHandler(ch)
    return logger
 
#通过静态成员方法来调用
class gaulog:
    logger = get_logger()
    
    @staticmethod
    def debug(msg):
        gaulog.logger.debug(ORANGE + "[DEBUG]: " + str(msg) + "\033[m", stacklevel=2)
    
    @staticmethod
    def info(msg):
        gaulog.logger.info("[INFO]: " + str(msg), stacklevel=2)
    
    @staticmethod
    def okinfo(msg):
        gaulog.logger.info(GREEN + "[INFO]: " + str(msg) + "\033[m", stacklevel=2)
    
    @staticmethod
    def warning(msg):
        gaulog.logger.warning("\033[38;5;214m" + "[WARNING]: " + str(msg) + "\033[m", stacklevel=2)
    
    @staticmethod
    def error(msg):
        gaulog.logger.error(RED + "[ERROR]: " + str(msg) + "\033[m", stacklevel=2)
    
    @staticmethod
    def critical(msg):
        gaulog.logger.critical(RED + "[CRITICAL]: " + str(msg) + "\033[m", stacklevel=2)
