def init_sys_path():
    import os
    import sys
    depth = 2
    path = os.path.dirname(os.path.realpath(__file__))
    for d in range(depth):
        path = os.path.join(path, os.pardir)
    proj_dir = os.path.abspath(path)
    sys.path.append(os.path.join(proj_dir, 'init'))
    import initpath
    initpath.platform_init_path(proj_dir)
