import multiprocessing
import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
print("CONFIG DIR IS :", CONFIG_DIR)
DIR_PATH = os.path.abspath(CONFIG_DIR + '/../') + "/"
print("DIR_Path is:",DIR_PATH)

GEM_PATH = DIR_PATH + "/../"

DYN_GEM_PATH = DIR_PATH + "/../"

NUM_CORES = multiprocessing.cpu_count()

REMOTE_DIR_PATH = "/run/user/1002/gvfs/sftp:host=alpha/home/mellers/"  # only used for evaluations

NODE2VEC_SNAP_DIR = DIR_PATH + "/snap/examples/node2vec/"
