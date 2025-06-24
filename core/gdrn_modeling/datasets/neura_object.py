import hashlib
import logging
import os
import os.path as osp
import sys
from pathlib import Path
cur_dir = Path(osp.dirname(osp.abspath(__file__))).parent.parent.parent
PROJ_ROOT = cur_dir
print("Project root: ", PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)

import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask, get_edge
from lib.utils.utils import dprint, iprint, lazy_property
from lib.vis_utils.image import grid_show
from lib.utils.setup_logger import setup_my_logger
from setproctitle import setproctitle
import torch
import matplotlib.pyplot as plt

import detectron2.data.datasets  # noqa # add pre-defined metadata
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import vis_image_mask_bbox_cv2
from lib.utils.time_utils import get_time_str
from lib.utils.utils import iprint
from core.utils.utils import get_emb_show
from core.utils.data_utils import read_image_mmcv
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from mmcv import Config
import yaml
import glob
import h5py
import json
from typing import List
from core.gdrn_modeling.datasets.data_loader_online import GDRN_Online_DatasetFromList
from lib.utils.config_utils import try_get_key


DATASETS_ROOT = osp.join(PROJ_ROOT, f"data/BOP_DATASETS/neura_objects") 
logger = logging.getLogger(__name__)
logger.info("Data root: ", DATASETS_ROOT)
NEURA_OBJECTS = ref.neura_object.objects # if neura_objects in the ref file is edited, please delete pycache and setup the module again

NEURA_CFG = dict(
    neura_train=dict(
        name=f"neura_train",
        mode = "train",
        objs=NEURA_OBJECTS,
        models_root=osp.join(DATASETS_ROOT, "models"),
        dataset_root=DATASETS_ROOT,
        scale_to_meter=0.001,
        with_masks=True,
        with_depth=True,
        height=480,
        width=640,
        cache_dir=osp.join(DATASETS_ROOT, ".cache"),
        use_cache=False,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="neura_object",  # make sure this has the same name as the file in reference. i.e. ref/neura_object.py as this will be used to reference the metadata in detectron2
    ),
    neura_val=dict(
        name=f"neura_val",
        mode = "val",
        objs=NEURA_OBJECTS,
        models_root=osp.join(DATASETS_ROOT, "models"),
        dataset_root=DATASETS_ROOT,
        scale_to_meter=0.001,
        with_masks=True,
        with_depth=True,
        height=480,
        width=640,
        cache_dir=osp.join(DATASETS_ROOT, ".cache"),
        use_cache=False,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="neura_object",  # make sure this has the same name as the file in reference. i.e. ref/neura_object.py as this will be used to reference the metadata in detectron2
    ),
    neura_test=dict(
        name=f"neura_test",
        mode = "test",
        objs=NEURA_OBJECTS,
        models_root=osp.join(DATASETS_ROOT, "models"),
        dataset_root=DATASETS_ROOT,
        scale_to_meter=0.001,
        with_masks=True,
        with_depth=True,
        height=480,
        width=640,
        cache_dir=osp.join(DATASETS_ROOT, ".cache"),
        use_cache=False,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="neura_object",  # make sure this has the same name as the file in reference. i.e. ref/neura_object.py as this will be used to reference the metadata in detectron2
    )
)

def get_available_datasets():
    """
    Returns the config files used for train / test of neura object
    """
    return list(NEURA_CFG.keys())

class NeuraPoseDataset:
    def __init__(self, data_cfg):
        """         
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects
        self.mode = data_cfg["mode"] # mode for train / val / test
        self.dataset_root = data_cfg['dataset_root']
        assert osp.exists(self.dataset_root), self.dataset_root

        self.models_root = data_cfg["models_root"]
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001
        self.model_path = sorted(Path(self.models_root).rglob("*.ply"))
        assert len(self.model_path) != 0, "unable to find {obj_name}.ply at {self.models_root}"

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        ##################################################

        self.cat_ids = [cat_id for cat_id, obj_name in ref.neura_object.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # 0 based labels
        print("Neura Object cat2label:", self.cat2label)
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################
        
        
    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files

        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(
            self.cache_dir,
            "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name),
        )
        

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []
        logger.info(f"Loaded {self.data_cfg['mode']} scenes")
        mode_path = osp.join(self.dataset_root, self.mode)
        with h5py.File(osp.join(mode_path, 'per_object_annotations.h5'), "r") as file:
            # Access a specific dataset
            num_train_images = len(file['color_path'])
            
            for i in tqdm(range(num_train_images)):
                rgb_path = file['color_path'][i][0].decode('utf-8')
                assert osp.exists(osp.join(mode_path,rgb_path)), osp.join(mode_path, rgb_path)
                
                obj_to_cam = file["object_to_cam"][i] # camera extrinsic matrix, obj w.r.t. camera
                depth_path = file['depth_path'][i][0].decode('utf-8')
                mask_path = file["mask_path"][i][0].decode('utf-8')
                K = file["K"][i]
                R = obj_to_cam[:3,:3]
                quat = mat2quat(R)
                t = obj_to_cam[:3, 3] # translation matrix, why they divide by 1000? TODO: Check out why divide by 1000
                obj_id = file["class_id"][i][0]
                proj = (K @ t.T).T
                proj = proj[:2] / proj[2]
                depth_factor = 1000.0 / file["depth_scale"][i]
                bbox_2d = np.asarray([file['bounding_box'][i]]).reshape(1,4) # make sure the shape is correct for the bbox to be converted
                x, y, w, h = bbox_2d[0]
                # mask
                mask = mmcv.imread(osp.join(mode_path, mask_path), 'unchanged')
                segmentation_index = file["segmentation_index"][i]
                mask_single_object = (mask == segmentation_index) # boolean mask for a single object
                mask_single_object = mask_single_object.astype('bool')
                mask = mask.astype('bool')
                # very important to filter all bad segmentation mask before registering the dataset for train/val/test
                # as this improves training stability
                if np.sum(mask_single_object) < 100 or np.sum(mask_single_object) / (self.height * self.width) > 0.7:
                    self.num_instances_without_valid_segmentation += 1
                    continue
                
                # very important to filter all bad bounding boxes before registering the dataset for train/val/test
                # as this improves training stability
                if self.filter_invalid: # filter invalid bounding boxes
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue               
                
                mask = binary_mask_to_rle(mask, compressed=True)
                mask_single_object = binary_mask_to_rle(mask_single_object, compressed=True)
                model_info = self.models_info[obj_id] # double check on the model path
                
                record = {
                    "id": i,
                    "dataset_name": self.name,
                    "file_name": osp.join(mode_path, rgb_path), 
                    "depth_file": osp.join(mode_path, depth_path), 
                    "mask_path": osp.join(mode_path, mask_path), 
                    "height": self.height,
                    "width": self.width,
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr",
                    "annotations": [{
                        "category_id": self.cat2label[obj_id], 
                        "bbox": np.asarray(bbox_2d),
                        "bbox_obj": bbox_2d,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": obj_to_cam,
                        "quat": quat,
                        "trans": t ,
                        "centroid_2d": proj, # absolute (cx, cy)
                        "segmentation": mask, # let mask_visible = mask_full
                        "mask_full": mask,
                        "visib_fract": 1, # Assume that visible fraction is 1 for now, all objects are visible. It is a filter to remove objects that are too small.
                        "model_info": model_info,
                        "model_path": self.model_path[self.cat2label[obj_id]],
                        "bbox3d_and_center": self.models[self.cat2label[obj_id]]['bbox3d_and_center'] 
                    }]
                }

                dataset_dicts.append(record)

        logger.info(f"loaded neura object dataset in {time.perf_counter() - t_start}s")

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        print(f"Removed {self.num_instances_without_valid_segmentation} data due to invalid segmentation")
        print(f"Removed {self.num_instances_without_valid_box} data due to invalid bbox")

        logger.info("Dumped dataset_dicts to {}".format(cache_path))


        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "objects_info.yaml")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # our own dataset, key = int
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.neura_object.obj2id[obj_name]:06d}",
                    f"obj_{ref.neura_object.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3
    
def get_neura_object_metadata(object_names: str, ref_key: str):
    """
    Gets the object metadata
    """
    cur_sym_info = {}
    data_ref = ref.__dict__[ref_key]
    loaded_models_info = data_ref.get_models_info() # load symmetry information from models/objects_info.yaml

    for i, obj_name in enumerate(object_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[obj_id]

        if 'symmetries_discrete' or 'symmetries_continuous':
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)

        else:
            sym_info = None

        cur_sym_info[obj_id] = sym_info

    meta = {"thing_classes": object_names, "sym_infos": cur_sym_info}
    return meta

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    neura_cfgs = NEURA_CFG
    if name in neura_cfgs:
        used_cfg = neura_cfgs[name]
    else:
        assert data_cfg is not None, f"Neura object dataset is not registered"

    DatasetCatalog.register(name, NeuraPoseDataset(used_cfg))
    dprint("register dataset: {}".format(name))

    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_neura_object_metadata(used_cfg["objs"], used_cfg["ref_key"]),
    )

# visualize
def test_vis(base_fp: str):

    dset_name = "neura_test"
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    index = 0
    for d in dicts:
        file_name = d['file_name'].split('/')[-1].split('.')[0] # rgb/scene_000000_frame_000000.jpg -> scene_000000_frame
        img = read_image_mmcv(osp.join(base_fp, 'train', 'rgb', f"{file_name}.jpg"), format="BGR") # testing purposes, hard coded string
        depth = mmcv.imread(osp.join(base_fp, 'train', 'depth', f"{file_name}.png"), "unchanged") / d['depth_factor'] # 10000.

        imH, imW = img.shape[:2]
        annos = d["annotations"][0]

        masks = np.asarray([cocosegm2mask(annos['segmentation'], imH, imW)])
        bboxes = annos['bbox_obj']
        bbox_modes = [annos['bbox_mode']]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box[np.newaxis,:], box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        bboxes_xyxy = bboxes_xyxy.squeeze()
        # kpts_3d_list = np.asarray([annos['model']])
        quats = np.asarray([annos['quat']])
        transes = np.asarray([annos['trans']])
        Rs = np.asarray([quat2mat(quats[0])])
        cat_id = annos['category_id'] # Assume single class for now
        K = np.asarray(d["cam"])
        # kpts_2d = np.asarray([misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)])
        labels = [objs[cat_id]] # np.asarray([objs[cat_id] for cat_id in cat_ids])

        img_vis = vis_image_mask_bbox_cv2(img, masks, bboxes=bboxes_xyxy[np.newaxis,:], labels=labels)

        grid_show(
            [
                img[:, :, [2, 1, 0]],
                img_vis[:, :, [2, 1, 0]],
                # img_vis_kpts2d[:, :, [2, 1, 0]],
                depth,
                # xyz ignore for now, as we are going to render online later,
                # diff_mask_xyz,
                # xyz_crop_show,
                # img_xyz[:, :, [2, 1, 0]],
                # img_xyz_crop[:, :, [2, 1, 0]],
                # img_vis_crop,
            ],
            [
                "img",
                "vis_img",
                # "img_vis_kpts2d",
                "depth",
                # "diff_mask_xyz",
                # "xyz_crop_show",
                # "img_xyz",
                # "img_xyz_crop",
                # "img_vis_crop",
            ],
            row=1,
            col=3,
        )
        index += 1
        if index == 10:
            break
    
def get_neura_object_metadata(object_names: str, ref_key: str):
    """
    Gets the object metadata
    """
    cur_sym_info = {}
    data_ref = ref.__dict__[ref_key]
    loaded_models_info = data_ref.get_models_info() # load symmetry information from models/objects_info.yaml

    for i, obj_name in enumerate(object_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[obj_id]

        if 'symmetries_discrete' or 'symmetries_continuous':
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)

        else:
            sym_info = None

        cur_sym_info[obj_id] = sym_info

    meta = {"thing_classes": object_names, "sym_infos": cur_sym_info}
    return meta

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    neura_cfgs = NEURA_CFG
    if name in neura_cfgs:
        used_cfg = neura_cfgs[name]
    else:
        assert data_cfg is not None, f"Neura object dataset is not registered"

    DatasetCatalog.register(name, NeuraPoseDataset(used_cfg))
    dprint("register dataset: {}".format(name))

    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_neura_object_metadata(used_cfg["objs"], used_cfg["ref_key"]),
    )

# visualize
def test_vis(base_fp: str):

    dset_name = "neura_train"
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    index = 0
    for d in dicts:
        file_name = d['file_name'].split('/')[-1].split('.')[0] # rgb/scene_000000_frame_000000.jpg -> scene_000000_frame
        img = read_image_mmcv(osp.join(base_fp, 'train', 'rgb', f"{file_name}.jpg"), format="BGR") # testing purposes, hard coded string
        depth = mmcv.imread(osp.join(base_fp, 'train', 'depth', f"{file_name}.png"), "unchanged") / d['depth_factor'] # 10000.

        imH, imW = img.shape[:2]
        annos = d["annotations"][0]

        masks = np.asarray([cocosegm2mask(annos['segmentation'], imH, imW)])
        bboxes = annos['bbox_obj']
        bbox_modes = [annos['bbox_mode']]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box[np.newaxis,:], box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        bboxes_xyxy = bboxes_xyxy.squeeze()
        # kpts_3d_list = np.asarray([annos['model']])
        quats = np.asarray([annos['quat']])
        transes = np.asarray([annos['trans']])
        Rs = np.asarray([quat2mat(quats[0])])
        cat_id = annos['category_id'] # Assume single class for now
        K = np.asarray(d["cam"])
        # kpts_2d = np.asarray([misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)])
        labels = [objs[cat_id]] # np.asarray([objs[cat_id] for cat_id in cat_ids])

        img_vis = vis_image_mask_bbox_cv2(img, masks, bboxes=bboxes_xyxy[np.newaxis,:], labels=labels)

        grid_show(
            [
                img[:, :, [2, 1, 0]],
                img_vis[:, :, [2, 1, 0]],
                # img_vis_kpts2d[:, :, [2, 1, 0]],
                depth,
                # xyz ignore for now, as we are going to render online later,
                # diff_mask_xyz,
                # xyz_crop_show,
                # img_xyz[:, :, [2, 1, 0]],
                # img_xyz_crop[:, :, [2, 1, 0]],
                # img_vis_crop,
            ],
            [
                "img",
                "vis_img",
                # "img_vis_kpts2d",
                "depth",
                # "diff_mask_xyz",
                # "xyz_crop_show",
                # "img_xyz",
                # "img_xyz_crop",
                # "img_vis_crop",
            ],
            row=1,
            col=3,
        )
        index += 1
        if index == 10:
            break


if __name__ == "__main__":
    """Test the  dataset loader in a notebook environment.

    Usage:
        python -m this_module bin_sort_tube
    """
    import sys
    from core.gdrn_modeling.main_gdrn import setup
    config_fp = "../../../configs/gdrn/neura_object/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura_object.py"
    sys.argv = ['ipykernel_launcher.py', '--config-file', config_fp]

    # Create the parser and parse the arguments
    parser = my_default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)
    print(f"Systems args {vars(args)}")

    test_vis(os.path.join(DATASETS_ROOT))
