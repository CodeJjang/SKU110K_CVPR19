import logging
import tqdm
import sys


class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm.
    Fixes logging while tqdm prints its progress bar.
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s: %(levelname).1s %(module)s.py %(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.NOTSET,
        stream=DummyTqdmFile(sys.stderr))
