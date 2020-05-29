import logging


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s: %(levelname).1s %(module)s.py %(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.NOTSET)
