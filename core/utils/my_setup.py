import logging
import os.path as osp
import sys
from pathlib import Path

# Import PROJ_ROOT from ref.neura_object as the single source of truth
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, str(Path(cur_dir).parent.parent))
import ref.neura_object


def get_project_root():
    return str(ref.neura_object.PROJ_ROOT)


PROJ_ROOT = get_project_root()


def get_data_root():
    proj_root = get_project_root()
    return osp.join(proj_root, "datasets")


DATA_ROOT = get_data_root()


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print
    if not is_master:
        logging.getLogger("core").setLevel("WARN")
        logging.getLogger("d2").setLevel("WARN")
        logging.getLogger("lib").setLevel("WARN")
        logging.getLogger("my").setLevel("WARN")

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
