# compute fps (farthest point sampling) for models
import os.path as osp
import sys
from tqdm import tqdm
from pathlib import Path

# Import PROJ_ROOT from ref.neura_object as the single source of truth
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, str(Path(cur_dir).parent.parent.parent.parent))
import ref.neura_object
PROJ_ROOT = str(ref.neura_object.PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)

import mmcv
from lib.pysixd import inout, misc
import ref
from core.utils.data_utils import get_fps_and_center

ref_key = "neura_object"
data_ref = ref.__dict__[ref_key]

model_dir = data_ref.model_dir
id2obj = data_ref.id2obj


def main():
    vertex_scale = 0.001
    fps_dict = {}
    for obj_id in tqdm(id2obj):
        print(obj_id)
        # Find the specific model path for this object ID
        model_paths = [str(i) for i in sorted(Path(model_dir).rglob(f"obj_{obj_id:06d}.ply"))]
        if not model_paths:
            print(f"Warning: No model found for object ID {obj_id}")
            continue
        model_path = model_paths[0]  # Use the first matching model path
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        fps_dict[str(obj_id)] = {}
        fps_dict[str(obj_id)]["fps4_and_center"] = get_fps_and_center(model["pts"], num_fps=4, init_center=True)
        fps_dict[str(obj_id)]["fps8_and_center"] = get_fps_and_center(model["pts"], num_fps=8, init_center=True)
        fps_dict[str(obj_id)]["fps12_and_center"] = get_fps_and_center(model["pts"], num_fps=12, init_center=True)
        fps_dict[str(obj_id)]["fps16_and_center"] = get_fps_and_center(model["pts"], num_fps=16, init_center=True)
        fps_dict[str(obj_id)]["fps20_and_center"] = get_fps_and_center(model["pts"], num_fps=20, init_center=True)
        fps_dict[str(obj_id)]["fps32_and_center"] = get_fps_and_center(model["pts"], num_fps=32, init_center=True)
        fps_dict[str(obj_id)]["fps64_and_center"] = get_fps_and_center(model["pts"], num_fps=64, init_center=True)
        fps_dict[str(obj_id)]["fps128_and_center"] = get_fps_and_center(model["pts"], num_fps=128, init_center=True)
        fps_dict[str(obj_id)]["fps256_and_center"] = get_fps_and_center(model["pts"], num_fps=256, init_center=True)

    save_path = osp.join(model_dir, "fps_points.pkl")
    mmcv.dump(fps_dict, save_path)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
