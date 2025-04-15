# import mmcv
# import os.path as osp
# import numpy as np
# import sys
# from tqdm import tqdm
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.structures import BoxMode
# import torch
# import gc
# from itertools import islice
# PROJ_ROOT = "/home/jeremy.ong/Desktop/experiments/pose_estimation/gdrnpp_bop2022/gdrnpp_bop2022/"

# cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, PROJ_ROOT)

# from lib.vis_utils.colormap import colormap
# from lib.utils.mask_utils import cocosegm2mask, get_edge
# from core.utils.data_utils import read_image_mmcv
# from core.gdrn_modeling.datasets.dataset_factory import register_datasets
# from transforms3d.quaternions import quat2mat
# from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
# from pathlib import Path
# import ref
# import random

# def process_image(d, preds, ren, image_tensor, seg_tensor, vis_dir, score_thr):
#     """Process a single image"""
#     K = d["cam"]
#     file_name = d["file_name"]
#     img = read_image_mmcv(file_name, format="BGR")

#     imH, imW = img.shape[:2]
#     annos = d["annotations"]
#     masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
#     bboxes = [anno["bbox"] for anno in annos]
#     bbox_modes = [anno["bbox_mode"] for anno in annos]
#     bboxes_xyxy = np.array(
#         [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
#     )
#     quats = [anno["quat"] for anno in annos]
#     transes = [anno["trans"] for anno in annos]
#     Rs = [quat2mat(quat) for quat in quats]
#     cat_ids = [anno["category_id"] for anno in annos]

#     est_Rs = []
#     est_ts = []
#     gt_Rs = []
#     gt_ts = []
#     labels = []

#     # Extract scene_im_id from file_name
#     try:
#         # Extract scene_im_id from file_name (format: scene_XXXXXX_frame_XXXXXX_XXXXXX.jpg)
#         parts = file_name.split('/')[-1].split('_')
#         if len(parts) >= 2 and parts[0] == 'scene':
#             scene_im_id = int(parts[1])
#             print(f"Extracted scene_im_id: {scene_im_id} from {file_name}")
#         else:
#             print(f"Could not extract scene_im_id from {file_name}, using file_name as key")
#             scene_im_id = file_name
#     except Exception as e:
#         print(f"Error extracting scene_im_id from {file_name}: {str(e)}")
#         scene_im_id = file_name

#     # Check if we have predictions for this image
#     if isinstance(preds, dict) and "test" in preds:
#         # Use the test predictions
#         print(f"Using test predictions, found {len(preds['test'])} predictions")
#         for pred in preds["test"]:
#             if pred["score"] < score_thr:
#                 print(f"Prediction score {pred['score']} below threshold {score_thr}, skipping")
#                 continue
            
#             R_est = pred["R"]
#             t_est = pred["t"]
#             obj_id = pred["obj_id"]
            
#             # Find matching annotation
#             for anno_i, anno in enumerate(annos):
#                 if anno["category_id"] == obj_id:
#                     cat_id = 0  # Use 0 as default category ID
#                     labels.append(cat_id)
                    
#                     est_Rs.append(R_est)
#                     est_ts.append(t_est)
#                     gt_Rs.append(Rs[anno_i])
#                     gt_ts.append(transes[anno_i])
#                     break
#     elif isinstance(preds, dict) and scene_im_id in preds:
#         # Use scene-specific predictions
#         print(f"Found predictions for scene_im_id: {scene_im_id}")
#         image_preds = preds[scene_im_id]
#         print(f"Number of predictions: {len(image_preds)}")
        
#         for pred in image_preds:
#             if pred["score"] < score_thr:
#                 print(f"Prediction score {pred['score']} below threshold {score_thr}, skipping")
#                 continue
            
#             R_est = pred["R"]
#             t_est = pred["t"]
#             obj_id = pred["obj_id"]
            
#             # Find matching annotation
#             for anno_i, anno in enumerate(annos):
#                 if anno["category_id"] == obj_id:
#                     cat_id = 0  # Use 0 as default category ID
#                     labels.append(cat_id)
                    
#                     est_Rs.append(R_est)
#                     est_ts.append(t_est)
#                     gt_Rs.append(Rs[anno_i])
#                     gt_ts.append(transes[anno_i])
#                     break
#     else:
#         print(f"No predictions found for scene_im_id: {scene_im_id}")
        
#     im_gray = mmcv.bgr2gray(img, keepdim=True)
#     im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

#     gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
#     poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

#     if len(labels) > 0:  # Only render if there are valid predictions
#         ren.render(labels, poses, K=K, image_tensor=image_tensor, background=im_gray_3)
#         ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

#         for label, gt_pose, est_pose in zip(labels, gt_poses, poses):
#             ren.render([label], [gt_pose], K=K, seg_tensor=seg_tensor)
#             gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

#             ren.render([label], [est_pose], K=K, seg_tensor=seg_tensor)
#             est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

#             gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
#             est_edge = get_edge(est_mask, bw=3, out_channel=1)

#             ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))
#             ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

#         vis_im = ren_bgr
#         random_num = random.randint(0, 10000)
        
#         # Extract the base filename without extension
#         base_filename = file_name.split('/')[-1].split('.')[0]
        
#         # Create unique filenames for each image
#         save_path_0 = osp.join(vis_dir, f"{base_filename}_{random_num:06d}_vis0.png")
#         save_path = osp.join(vis_dir, f"{base_filename}_{random_num:06d}_vis1.png")
        
#         # Save camera matrix
#         K_file_path = osp.join(vis_dir, f"{base_filename}_{random_num:06d}_K.txt")
        
#         # Save all files
#         mmcv.imwrite(img, save_path_0)
#         mmcv.imwrite(vis_im, save_path)
#         np.savetxt(K_file_path, K, fmt='%.6f')
        
#         print(f"Saved visualization images: {save_path_0} and {save_path}")
#         print(f"Saved camera matrix to: {K_file_path}")

#     # Clean up memory
#     del img, masks, gt_poses, poses, im_gray, im_gray_3
#     if 'ren_bgr' in locals(): del ren_bgr
#     if 'vis_im' in locals(): del vis_im
#     torch.cuda.empty_cache()
#     gc.collect()

# def main():
#     global cat2label, objs  # Move all global declarations to the start of the function
    
#     DATASET_ROOT = osp.join(PROJ_ROOT,"data/BOP_DATASETS/neura_objects")
#     score_thr = 0.3
#     colors = colormap(rgb=False, maximum=255)

#     objs = ref.neura_object.objects
#     id2obj = ref.__dict__["neura_object"].id2obj
#     cat_ids = [cat_id for cat_id, obj_name in ref.neura_object.id2obj.items() if obj_name in objs]
#     cat2label = {v: i for i, v in enumerate(cat_ids)}
#     objects = list(id2obj.values())

#     width = 640
#     height = 480
#     tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
#     image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
#     seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

#     model_dir = osp.join(DATASET_ROOT, 'models')
#     model_paths = [str(f) for f in sorted(list(Path(model_dir).rglob("obj_*.ply")))]

#     ren = EGLRenderer(model_paths, vertex_scale=0.001, use_cache=False, width=width, height=height)

#     pred_path = osp.join(PROJ_ROOT, "output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/results.pkl")
#     vis_dir = osp.join(PROJ_ROOT, "output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/neura_pred_full")
#     mmcv.mkdir_or_exist(vis_dir)

#     # Check if predictions file exists
#     if not osp.exists(pred_path):
#         print(f"Error: Predictions file not found at {pred_path}")
#         return
        
#     # Load predictions
#     preds = mmcv.load(pred_path)
#     print(f"Loaded predictions from {pred_path}")
#     print(f"Type of predictions: {type(preds)}")
    
#     # Debug: Print structure of predictions
#     if isinstance(preds, dict):
#         print(f"Predictions keys: {list(preds.keys())}")
#         if len(preds) > 0:
#             sample_key = list(preds.keys())[0]
#             print(f"Sample key: {sample_key}")
#             print(f"Sample value type: {type(preds[sample_key])}")
#             if isinstance(preds[sample_key], list) and len(preds[sample_key]) > 0:
#                 print(f"Sample prediction: {preds[sample_key][0]}")
    
#     dataset_name = "neura_test"
#     register_datasets([dataset_name])

#     meta = MetadataCatalog.get(dataset_name)
#     objs = meta.objs  # Update the global objs variable

#     # Get the dataset
#     dset_dicts = DatasetCatalog.get(dataset_name)
#     print(f"Dataset type: {type(dset_dicts)}")
    
#     # Process all images directly
#     if isinstance(dset_dicts, list):
#         print(f"Dataset is a list with {len(dset_dicts)} entries")
#         for i, d in enumerate(tqdm(dset_dicts, desc="Processing images")):
#             print(f"Processing image {i+1}/{len(dset_dicts)}: {d['file_name']}")
#             process_image(d, preds, ren, image_tensor, seg_tensor, vis_dir, score_thr)
            
#             # Clean up after each image
#             torch.cuda.empty_cache()
#             gc.collect()
#     else:
#         print(f"Dataset is not a list, type: {type(dset_dicts)}")
#         # Try to convert to a list if possible
#         if isinstance(dset_dicts, dict):
#             print("Converting dictionary dataset to list")
#             all_dicts = []
#             for scene_id, scene_dicts in dset_dicts.items():
#                 all_dicts.extend(scene_dicts)
#             dset_dicts = all_dicts
#             print(f"Converted to list with {len(dset_dicts)} entries")
            
#             # Process all images directly
#             for i, d in enumerate(tqdm(dset_dicts, desc="Processing images")):
#                 print(f"Processing image {i+1}/{len(dset_dicts)}: {d['file_name']}")
#                 process_image(d, preds, ren, image_tensor, seg_tensor, vis_dir, score_thr)
                
#                 # Clean up after each image
#                 torch.cuda.empty_cache()
#                 gc.collect()
#         else:
#             print("Cannot process dataset, unsupported type")

# if __name__ == "__main__":
#     main()
import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
PROJ_ROOT = "/home/jeremy.ong/Desktop/experiments/pose_estimation/gdrnpp_bop2022/gdrnpp_bop2022/"

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
from lib.vis_utils.colormap import colormap
import ref
from lib.utils.mask_utils import cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from pathlib import Path
import random

score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

id2obj = ref.__dict__["neura_object"].id2obj
objects = list(id2obj.values())
objs = ref.neura_object.objects
cat_ids = [cat_id for cat_id, obj_name in ref.neura_object.id2obj.items() if obj_name in objs]
cat2label = {v: i for i, v in enumerate(cat_ids)}
print("Cat2label:", cat2label)
width = 640
height = 480
tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = ref.neura_object.model_dir

model_paths = sorted([str(f) for f in list(Path(model_dir).rglob("obj_*.ply"))])

ren = EGLRenderer(model_paths, vertex_scale=0.001, use_cache=True, width=width, height=height)

# NOTE:
pred_path = osp.join(PROJ_ROOT, 'output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/results_list.pkl')

vis_dir = osp.join(PROJ_ROOT, "output/gdrn/neura/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura/inference_model_final/neura_test/neura_vis_gt_pred_full")
mmcv.mkdir_or_exist(vis_dir)

print(pred_path)
preds = mmcv.load(pred_path)

dataset_name = "neura_test"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
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
        if score < score_thr:
            continue
        
        est_Rs.append(R_est)
        est_ts.append(t_est)
        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])

    im_gray = mmcv.bgr2gray(img, keepdim=True)
    im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
    poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

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
        save_path_0 = osp.join(vis_dir, "{}_{:06d}_vis0.png".format(scene_im_id, random_num))
        mmcv.imwrite(img, save_path_0)

        save_path = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_im_id, random_num))
        mmcv.imwrite(vis_im, save_path)

    # if True:
    #     # grid_show([img[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    #     # im_show = cv2.hconcat([img, vis_im, vis_im_add])
    #     im_show = cv2.hconcat([img, vis_im])
    #     cv2.imshow("im_est", im_show)
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit

# ffmpeg -r 5 -f image2 -s 1920x1080 -pattern_type glob -i "./lmo_vis_gt_pred_full_video/*.png" -vcodec libx264 -crf 25  -pix_fmt yuv420p lmo_vis_video.mp4
