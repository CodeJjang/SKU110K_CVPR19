import logging
import os
import sys
import platform
import ntpath

__author__ = 'roeiherz'

FILE_EXISTS_ERROR = (17, 'File exists')

IMG_FOLDER = 'images'
ANNOTATION_FOLDER = 'annotations'
DEBUG_MODE = False  # 'ubuntu' not in os.environ['HOME']
if DEBUG_MODE:
    IMG_FOLDER += '_debug'
    ANNOTATION_FOLDER += '_debug'


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                logging.error(e)
            else:
                raise

        logging.info('Created folder {0}'.format(path))

    return folder_missing


def root_dir():
    if platform.system() == 'Linux':
        return os.path.join(os.getenv('HOME'), 'Documents', 'SKU110K')
    elif platform.system() == 'Windows':
        return os.path.abspath('C:/Users/{}/Documents/SKU110K/'.format(os.getenv('username')))


def image_path():
    return os.path.join(root_dir(), IMG_FOLDER)


def annotation_path():
    return os.path.join(root_dir(), ANNOTATION_FOLDER)


def create_dirpath_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_path_fname(path):
    '''
    Extract basename from file path
    '''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def is_path_exists(dir_path):
    return os.path.exists(dir_path)


def create_dirpath_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def rm_dir(dir_path):
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)


def get_path_fname(path):
    '''
    Extract basename from file path
    '''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_last_folder(path):
    return os.path.basename(os.path.normpath(path))
