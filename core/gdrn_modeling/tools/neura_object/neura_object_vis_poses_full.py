import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import gc
from itertools import islice
PROJ_ROOT = "/home/jeremy.ong/Desktop/experiments/pose_estimation/gdrnpp_bop2022/gdrnpp_bop2022/"

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from pathlib import Path
import ref
import random

def process_batch(batch_dicts, preds, ren, image_tensor, seg_tensor, vis_dir, pred_index, K_tensor_kwargs, score_thr):
    """Process a batch of images"""
    current_file_name = ""
    counter = 0
    local_pred_index = pred_index
    
    for d in tqdm(batch_dicts, desc="Processing batch"):
        K = d["cam"]
        if current_file_name == d["file_name"]:
            counter += 1
        else:
            counter = 0

        file_name = d["file_name"]
        img = read_image_mmcv(file_name, format="BGR")
        file_name += f"_{counter}"

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
        cat_ids = [anno["category_id"] for anno in annos]

        est_Rs = []
        est_ts = []
        gt_Rs = []
        gt_ts = []
        labels = []

        for anno_i, anno in enumerate(annos):
            try:
                R_est = preds["test"][local_pred_index]["R"]
                t_est = preds["test"][local_pred_index]["t"]
                score = preds["test"][local_pred_index]["score"]
            except:
                continue
            if score < score_thr:
                continue
            

            cat_id = 0 # cat2label[preds["test"][local_pred_index]['obj_id']]
            labels.append(cat_id)
            
            est_Rs.append(R_est)
            est_ts.append(t_est)
            gt_Rs.append(Rs[anno_i])
            gt_ts.append(transes[anno_i])
            local_pred_index += 1
            
        im_gray = mmcv.bgr2gray(img, keepdim=True)
        im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

        gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
        poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

        if len(labels) > 0:  # Only render if there are valid predictions
            ren.render(labels, poses, K=K, image_tensor=image_tensor, background=im_gray_3)
            ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

            for label, gt_pose, est_pose in zip(labels, gt_poses, poses):
                ren.render([label], [gt_pose], K=K, seg_tensor=seg_tensor)
                gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

                ren.render([label], [est_pose], K=K, seg_tensor=seg_tensor)
                est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

                gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
                est_edge = get_edge(est_mask, bw=3, out_channel=1)

                ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))
                ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

            vis_im = ren_bgr
            random_num = random.randint(0, 10000)
            
            save_path_0 = osp.join(vis_dir, f"{file_name.split('/')[-1].split('.')[0]}_{random_num:06d}_vis0.png")
            save_path = osp.join(vis_dir, f"{file_name.split('/')[-1].split('.')[0]}_{random_num:06d}_vis1.png")
            
            mmcv.imwrite(img, save_path_0)
            mmcv.imwrite(vis_im, save_path)

        # Clean up memory
        del img, masks, gt_poses, poses, im_gray, im_gray_3
        if 'ren_bgr' in locals(): del ren_bgr
        if 'vis_im' in locals(): del vis_im
        torch.cuda.empty_cache()
        gc.collect()

    return local_pred_index

def main():
    global cat2label, objs  # Move all global declarations to the start of the function
    
    DATASET_ROOT = osp.join(PROJ_ROOT,"data/BOP_DATASETS/neura_objects")
    score_thr = 0.3
    colors = colormap(rgb=False, maximum=255)

    objs = ref.neura_object.objects
    id2obj = ref.__dict__["neura_object"].id2obj
    cat_ids = [cat_id for cat_id, obj_name in ref.neura_object.id2obj.items() if obj_name in objs]
    cat2label = {v: i for i, v in enumerate(cat_ids)}
    objects = list(id2obj.values())

    width = 640
    height = 480
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    model_dir = osp.join(DATASET_ROOT, 'models')
    model_paths = [str(f) for f in sorted(list(Path(model_dir).rglob("obj_*.ply")))]

    ren = EGLRenderer(model_paths, vertex_scale=0.001, use_cache=False, width=width, height=height)

    pred_path = osp.join(PROJ_ROOT, "output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/results.pkl")
    vis_dir = osp.join(PROJ_ROOT, "output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/neura_pred_full")
    mmcv.mkdir_or_exist(vis_dir)

    preds = mmcv.load(pred_path)
    dataset_name = "neura_test"
    register_datasets([dataset_name])

    meta = MetadataCatalog.get(dataset_name)
    objs = meta.objs  # Update the global objs variable

    dset_dicts = DatasetCatalog.get(dataset_name)
    
    # Process in batches
    BATCH_SIZE = 100  # Adjust this based on your available RAM
    pred_index = 0
    
    for i in range(0, len(dset_dicts), BATCH_SIZE):
        batch = list(islice(dset_dicts, i, i + BATCH_SIZE))
        pred_index = process_batch(batch, preds, ren, image_tensor, seg_tensor, vis_dir, pred_index, tensor_kwargs, score_thr)
        
        # Clean up after each batch
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
