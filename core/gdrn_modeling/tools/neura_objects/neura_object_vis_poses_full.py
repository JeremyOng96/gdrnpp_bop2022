import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
from pathlib import Path

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, str(Path(cur_dir).parent.parent.parent.parent))
import ref.neura_object
PROJ_ROOT = str(ref.neura_object.PROJ_ROOT)

from lib.vis_utils.colormap import colormap
import ref
from lib.utils.mask_utils import cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from pathlib import Path
import random
import argparse

class Visualize:
    def __init__(self, args):
        super().__init__()
        self.score_threshold = args.score_thr
        self.pred_path = args.pred_path
        self.output_path = args.output_path

        # default parameters
        self.colors = colormap(rgb=False, maximum=255)
        self.id2obj = ref.__dict__["neura_object"].id2obj
        self.objects = list(self.id2obj.values())
        self.objs = ref.neura_object.objects
        self.cat_ids = [cat_id for cat_id, obj_name in self.id2obj.items() if obj_name in self.objects]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}
        self.model_dir = ref.neura_object.model_dir
        self.model_paths = sorted([str(f) for f in list(Path(self.model_dir).rglob("obj_*.ply"))])

        self.width = 640
        self.height = 480
        self.tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
        self.image_tensor = torch.empty((self.height, self.width, 4), **self.tensor_kwargs).detach()
        self.seg_tensor = torch.empty((self.height, self.width, 4), **self.tensor_kwargs).detach()
        
        self.init_renderer()

    def init_renderer(self):
        self.ren = EGLRenderer(self.model_paths, vertex_scale=0.001, use_cache=False, width=self.width, height=self.height)


    def visualize(self):
        mmcv.mkdir_or_exist(self.output_path)
        preds = mmcv.load(self.pred_path)

        dataset_name = "neura_test"
        register_datasets([dataset_name])
        meta = MetadataCatalog.get(dataset_name)
        objs = meta.objs

        dset_dicts = DatasetCatalog.get(dataset_name)
        for index, d in tqdm(enumerate(dset_dicts)):
            K = d["cam"]
            file_name = d["file_name"]
            img = read_image_mmcv(file_name, format="BGR")

            scene_im_id_split = d["file_name"].split("/")[-1]
            scene_im_id, ext = osp.splitext(scene_im_id_split)

            imH, imW = img.shape[:2]
            annos = d["annotations"]

            masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
            bboxes = [anno["bbox"] for anno in annos]
            bbox_modes = [anno["bbox_mode"] for anno in annos]
            bboxes_xyxy = np.array(
                [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
            )
            quats = [anno["quat"] for anno in annos]
            transes = [anno["trans"] for anno in annos]
            Rs = [quat2mat(quat) for quat in quats]
            # 0-based label
            cat_ids = [anno["category_id"] for anno in annos]

            obj_names = [objs[cat_id] for cat_id in cat_ids]

            est_Rs = []
            est_ts = []

            gt_Rs = []
            gt_ts = []
            
            labels = [annos[0]['category_id']] # assume one object per image, cause of Neura format
            
            for anno_i, anno in enumerate(annos):
                # obj_name = obj_names[anno_i]
                try:
                    R_est = preds[index]["R"]
                    t_est = preds[index]["t"]
                    score = preds[index]["score"]
                except:
                    continue
                if score < self.score_threshold:
                    continue
                
                est_Rs.append(R_est)
                est_ts.append(t_est)
                gt_Rs.append(Rs[anno_i])
                gt_ts.append(transes[anno_i])

            im_gray = mmcv.bgr2gray(img, keepdim=True)
            im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

            gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
            poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

            self.ren.render(labels, poses, K=K, image_tensor=self.image_tensor, background=im_gray_3)
            ren_bgr = (self.image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

            for label, gt_pose, est_pose in zip(labels, gt_poses, poses):
                self.ren.render([label], [gt_pose], K=K, seg_tensor=self.seg_tensor)
                gt_mask = (self.seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

                self.ren.render([label], [est_pose], K=K, seg_tensor=self.seg_tensor)
                est_mask = (self.seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

                gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
                est_edge = get_edge(est_mask, bw=3, out_channel=1)

                ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))
                ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

                vis_im = ren_bgr
            
                random_num = random.randint(0, 10000)
                save_path_0 = osp.join(self.output_path, "{}_{:06d}_vis0.png".format(scene_im_id, random_num))
                mmcv.imwrite(img, save_path_0)

                save_path = osp.join(self.output_path, "{}_{:06d}_vis1.png".format(scene_im_id, random_num))
                mmcv.imwrite(vis_im, save_path)


def main(args):
    visualizer = Visualize(args)
    visualizer.visualize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose estimation")

    parser.add_argument(
        "--pred_path",
        default="",
        type=str,
        help="path to the prediction file"
    )

    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="path to the output directory"
    )
    
    parser.add_argument(
        "--score_thr",
        default=0.3,
        type=float,
        help="filter detection scores lower than threshold"
    )

    args = parser.parse_args()
    main(args)