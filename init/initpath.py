import sys
import os


def platform_init_path(proj_dir):
    sys.path.append(os.path.join(proj_dir, 'util/'))
