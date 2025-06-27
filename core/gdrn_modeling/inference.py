from loguru import logger as loguru_logger
import logging
import os
import random
os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from setproctitle import setproctitle
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import mmcv
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from mmcv import Config
import cv2
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite  # import LightningLite
from time import time
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_checkpoint import MyCheckpointer
import ref
import numpy as np

from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList, build_gdrn_test_loader
from core.utils.dataset_utils import trivial_batch_collator 
from core.gdrn_modeling.engine.engine_utils import batch_data_inference_roi, batch_data_test, batch_data
from core.gdrn_modeling.models import GDRN_double_mask
from pathlib import Path

import matplotlib.pyplot as plt
from core.gdrn_modeling.main_gdrn import Lite, setup
from lib.utils.mask_utils import get_edge

# visualize libraries
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer

# post processing libraries
from core.gdrn_modeling.engine.engine_utils import get_out_coor, get_out_mask
from lib.pysixd.pose_error import add, adi, arp_2d, re, te

logger = logging.getLogger("detectron2")

class Inference(Lite):
    def __init__(self, args, cfg):
        super().__init__(
            accelerator = 'gpu',
            strategy = args.strategy,
            devices = args.num_gpus,
            num_nodes = args.num_machines,
            precision = 16 if cfg.SOLVER.AMP.ENABLED else 32,
        )

        print("Configuration: ", cfg)
        self.objs = ref.neura_object.objects
        self._cpu_device = torch.device("cpu")
        self.cat_ids = [cat_id for cat_id, obj_name in ref.neura_object.id2obj.items() if obj_name in self.objs] 
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        
        # Add data_ref needed for process methods
        self.data_ref = ref.neura_object
        
        self.set_my_env(args, cfg)
        self.model, _ = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
        self.model_info = mmcv.load(args.model_info)
        self.cfg = cfg
        self.setup(self.model)
        MyCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(args.model_path, resume=False)
        logger.info(self.model)
        MetadataCatalog.get("neura_object").set(
            ref_key="neura_object",
            objs= self.objs,
        )
        self.vis = args.visualize
        self.verbose = args.verbose

    def predict(self, image, bbox, obj_id, camera_matrix, depth = None):
        h, w, c = image.shape       
        bbox_2d = np.asarray(bbox).reshape(1,4)
        bbox_2d = BoxMode.convert(bbox_2d, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        data = [{
            "dataset_name": "neura_object",
            "height": h,
            "width": w,
            "cam": camera_matrix,
            "image": image,
            "img_type": "img_real",
            "depth": depth,
            "depth_factor": 1000,
            "annotations": [{
                "category_id": self.cat2label[obj_id], # object_id starts from 1 to n and object_id != class_id
                "bbox": bbox_2d,
                "bbox_mode": BoxMode.XYXY_ABS,
                "model_info": self.model_info[self.cat2label[obj_id]],
            }]
        }]

        dataset = GDRN_DatasetFromList(self.cfg, split="inference", lst=data, flatten=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=None,
            collate_fn=trivial_batch_collator,
            pin_memory=False,
        )
        data = next(iter(data_loader))
        with torch.no_grad():
            batch = batch_data(self.cfg, data, renderer=None, phase='test', device='cuda')
            output = self.model(
                batch["roi_img"],
                sym_infos=batch.get("sym_info", None),
                roi_classes=batch["roi_cls"],
                roi_cams=batch["roi_cam"],
                roi_whs=batch["roi_wh"],
                roi_centers=batch["roi_center"],
                resize_ratios=batch["resize_ratio"],
                roi_coord_2d=batch.get("roi_coord_2d", None),
                roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
                roi_extents=batch.get("roi_extent", None),
            )
        
        refined_output = self.process_net_and_pnp(data, output)

        if self.vis:
            # Use refined pose for visualization
            data[0]["pred_rot"] = refined_output["R"]
            data[0]["pred_trans"] = refined_output["t"]
            image_visualized = self.visualize(data)

            if self.verbose:
                logger.info(f"Rotatoin matrix: {refined_output['R']}")
                logger.info(f"Translation vector: {refined_output['t']}")
                plt.figure()
                plt.imshow(image_visualized)
                plt.show()
                
            return refined_output["R"], refined_output["t"], image_visualized
        
        return refined_output["R"], refined_output["t"]
    
    def visualize(self, data):
        height, width = data[0]["height"], data[0]["width"]
        tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
        image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
        seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
        model_dir = ref.neura_object.model_dir
        model_paths = sorted([str(f) for f in list(Path(model_dir).rglob("obj_*.ply"))])
        ren = EGLRenderer(model_paths, vertex_scale=0.001, use_cache=False, width=width, height=height) # initialize renderer

        # renderer parameters
        K = data[0]["cam"].squeeze()
        img = data[0]["image"]
        img_gray = mmcv.bgr2gray(img, keepdim=True)
        img_gray_3 = np.concatenate([img_gray, img_gray, img_gray], axis=2)
        label = data[0]["annotations"][0]["category_id"]
        
        pred_rot = data[0]["pred_rot"].cpu().numpy() if torch.is_tensor(data[0]["pred_rot"]) else data[0]["pred_rot"]
        pred_trans = data[0]["pred_trans"].cpu().numpy() if torch.is_tensor(data[0]["pred_trans"]) else data[0]["pred_trans"]
        logger.info(f"pred_trans: {pred_trans}")
        pred_pose = np.hstack([pred_rot.squeeze(), pred_trans.reshape(3,1)])      
        
        logger.info(f"pred_pose: {pred_pose}")
        ren.render(label, pred_pose, K=K, image_tensor=image_tensor, background=img_gray_3)
        ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

        ren.render([label], [pred_pose], K=K, seg_tensor=seg_tensor)
        pred_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        pred_edge = get_edge(pred_mask, bw=3, out_channel=1)
        ren_bgr[pred_edge != 0] = np.array(mmcv.color_val("green"))
        vis_im = ren_bgr

        return vis_im

    def process_net_and_pnp(self, inputs, out_dict, pnp_type="iter"):
            """Initialize with network prediction (learned PnP) + iter PnP
            Args:
                inputs: the inputs to a model.
                    It is a list of dict. Each dict corresponds to an image and
                    contains keys like "height", "width", "file_name", "image_id", "scene_id".
                pnp_type: iter | ransac (use ransac+EPnP)
                outputs:
            """
            cfg = self.cfg
            net_cfg = cfg.MODEL.POSE_NET
            out_coor_x = out_dict["coor_x"].detach()
            out_coor_y = out_dict["coor_y"].detach()
            out_coor_z = out_dict["coor_z"].detach()
            out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
            out_xyz = out_xyz.to(self._cpu_device).numpy()

            out_mask = get_out_mask(cfg, out_dict["mask"].detach())
            out_mask = out_mask.to(self._cpu_device).numpy()

            out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
            out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

            for i, _input in enumerate(inputs):
                logger.info(f"inputs: {_input}")
                coord_2d = _input["roi_coord_2d"].cpu().numpy().squeeze().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"].cpu().numpy()
                im_W = _input["im_W"].cpu().numpy()
                K = _input["cam"].cpu().numpy().squeeze().copy()
                roi_label = _input["roi_cls"]
                xyz = out_xyz.squeeze().transpose(1,2,0).squeeze()
                mask = np.squeeze(out_mask)

                (img_points, model_points,) = self.get_img_model_points_with_coords2d(
                        mask,
                        xyz,
                        coord_2d,
                        im_H=im_H,
                        im_W=im_W,
                        extent=_input["roi_extent"].cpu().numpy().squeeze(),
                        mask_thr=net_cfg.GEO_HEAD.MASK_THR_TEST,
                    )
                
                rot_est = out_rots
                rot_est_net = out_rots.squeeze()
                trans_est_net = out_transes.squeeze()
                
                num_points = len(img_points)
                if num_points >= 4:
                    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
                    points_2d = np.ascontiguousarray(img_points.astype(np.float64))
                    points_3d = np.ascontiguousarray(model_points.astype(np.float64))
                    camera_matrix = K.astype(np.float64)

                    rvec0, _ = cv2.Rodrigues(rot_est_net)

                    if pnp_type in ["ransac", "ransac_rot"]:
                        points_3d = np.expand_dims(points_3d, 0)
                        points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est, _ = cv2.solvePnPRansac(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                            reprojectionError=3.0,  # default 8.0
                            iterationsCount=20,
                        )
                    else:  # iter PnP
                        # points_3d = np.expand_dims(points_3d, 0)  # uncomment for EPNP
                        # points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est = cv2.solvePnP(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                            # flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                        )
                    rot_est, _ = cv2.Rodrigues(rvec)
                    if pnp_type not in ["ransac_rot"]:
                        diff_t_est = te(t_est, trans_est_net)
                        if diff_t_est > 1:  # diff too large
                            logger.warning("translation error too large: {}".format(diff_t_est))
                            t_est = trans_est_net
                    else:
                        t_est = trans_est_net
                    pose_est = np.concatenate([rot_est, t_est.reshape((3, 1))], axis=-1)
                else:
                    logger.warning("num points: {}".format(len(img_points)))
                    pose_est_net = np.hstack([rot_est_net, trans_est_net.reshape(3, 1)])
                    pose_est = pose_est_net

                result = {
                    "cls_name": roi_label,
                    "R": pose_est[:3, :3],
                    "t": pose_est[:3, 3],
                }
                
                return result
            
    def get_img_model_points_with_coords2d(
        self, mask_pred_crop, xyz_pred_crop, coord2d_crop, im_H, im_W, extent, max_num_points=-1, mask_thr=0.5
    ):
        """
        from predicted crop_and_resized xyz, bbox top-left,
        get 2D-3D correspondences (image points, 3D model points)
        Args:
            mask_pred_crop: HW, predicted mask in roi_size
            xyz_pred_crop: HWC, predicted xyz in roi_size(eg. 64)
            coord2d_crop: HW2 coords 2d in roi size
            im_H, im_W
            extent: size of x,y,z
        """
        # [0, 1] --> [-0.5, 0.5] --> original
        xyz_pred_crop[:, :, 0] = (xyz_pred_crop[:, :, 0] - 0.5) * extent[0]
        xyz_pred_crop[:, :, 1] = (xyz_pred_crop[:, :, 1] - 0.5) * extent[1]
        xyz_pred_crop[:, :, 2] = (xyz_pred_crop[:, :, 2] - 0.5) * extent[2]

        coord2d_crop = coord2d_crop.copy()
        coord2d_crop[:, :, 0] = coord2d_crop[:, :, 0] * im_W
        coord2d_crop[:, :, 1] = coord2d_crop[:, :, 1] * im_H

        sel_mask = (
            (mask_pred_crop > mask_thr)
            & (abs(xyz_pred_crop[:, :, 0]) > 0.0001 * extent[0])
            & (abs(xyz_pred_crop[:, :, 1]) > 0.0001 * extent[1])
            & (abs(xyz_pred_crop[:, :, 2]) > 0.0001 * extent[2])
        )
        model_points = xyz_pred_crop[sel_mask].reshape(-1, 3)
        image_points = coord2d_crop[sel_mask].reshape(-1, 2)

        if max_num_points >= 4:
            num_points = len(image_points)
            max_keep = min(max_num_points, num_points)
            indices = [i for i in range(num_points)]
            random.shuffle(indices)
            model_points = model_points[indices[:max_keep]]
            image_points = image_points[indices[:max_keep]]
        return image_points, model_points


def main(args):
    cfg = setup(args)
    image = cv2.imread(str(Path(cur_dir).parent.parent / "data/BOP_DATASETS/neura_objects/test/rgb/scene_000002_frame_000000.jpg"))
    bbox = np.array([[331, 304,  61,  35]]) # XYWH format
    obj_id = 1
    camera_matrix = np.array([[449.9816,   0.0000, 321.1470],
                            [0.0000, 447.6302, 240.6461],
                            [0,            0,         1]])
    
    inference = Inference(args, cfg)
    inference.predict(image, bbox, obj_id, camera_matrix)

if __name__ == "__main__":
    parser = my_default_argument_parser()

    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="the strategy for parallel training: dp | ddp | ddp_spawn | deepspeed | ddp_sharded",
    )

    parser.add_argument(
        "--model_info",
        default=None,
        type=str,
        help="path to model info"
        )
    
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path to model weights"
    )

    parser.add_argument(
        "--visualize",
        default=False,
        type=bool,
        help="visualize the inference results"
    )

    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="verbose mode"
    )

    args = parser.parse_args()
    main(args)