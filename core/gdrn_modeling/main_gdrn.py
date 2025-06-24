from loguru import logger as loguru_logger
import logging
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from setproctitle import setproctitle
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# from torch.nn.parallel import DistributedDataParallel

# from detectron2.engine import launch
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from mmcv import Config
import cv2
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite  # import LightningLite

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from core.utils import my_comm as comm

from lib.utils.utils import iprint
from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str
import ref
import numpy as np

from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import get_renderer
from core.gdrn_modeling.engine.engine import GDRN_Lite
from core.gdrn_modeling.models import (
    GDRN,
    GDRN_no_region,
    GDRN_cls,
    GDRN_cls2reg,
    GDRN_double_mask,
    GDRN_Dstream_double_mask,
)  # noqa

from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList, build_gdrn_test_loader
from core.utils.dataset_utils import trivial_batch_collator 
from core.gdrn_modeling.engine.engine_utils import batch_data_inference_roi, batch_data_test

logger = logging.getLogger("detectron2")

def setup(args):
    """Create configs and perform basic setups."""
    cfg = Config.fromfile(args.config_file)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # NOTE: check if need to set OUTPUT_DIR automatically
    if cfg.OUTPUT_DIR.lower() == "auto":
        cfg.OUTPUT_DIR = osp.join(
            cfg.OUTPUT_ROOT,
            osp.splitext(args.config_file)[0].split("configs/")[1],
        )
        iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

    if cfg.get("EXP_NAME", "") == "":
        setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
    else:
        setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanted configs in detectron2
    # ---------------------------------------------------------
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    bs_ref = cfg.SOLVER.get("REFERENCE_BS", cfg.SOLVER.IMS_PER_BATCH)  # nominal batch size
    if bs_ref <= cfg.SOLVER.IMS_PER_BATCH:
        bs_ref = cfg.SOLVER.REFERENCE_BS = cfg.SOLVER.IMS_PER_BATCH
        # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
        # all-reduce operation is carried out during loss.backward().
        # Thus, there would be redundant all-reduce communications in a accumulation procedure,
        # which means, the result is still right but the training speed gets slower.
        # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
        # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
        accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate loss before optimizing
    else:
        accumulate_iter = 1
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        if accumulate_iter > 1:
            if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
                cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
                    cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
                )  # scale weight_decay
    if accumulate_iter > 1:
        cfg.SOLVER.WEIGHT_DECAY *= cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
    # -------------------------------------------------------------------------
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
    # register datasets
    register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        if cfg.TEST.USE_PNP:
            # NOTE: need to keep _test at last
            exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
        else:
            exp_id += "_test"
    cfg.EXP_ID = exp_id
    cfg.RESUME = args.resume
    ####################################
    # cfg.freeze()
    return cfg



class Lite(GDRN_Lite):
    def set_my_env(self, args, cfg):
        my_default_setup(cfg, args)  # will set os.environ["PYTHONHASHSEED"]
        seed_everything(int(os.environ["PYTHONHASHSEED"]), workers=True)
        setup_for_distributed(is_master=self.is_global_zero)
        
    def run(self, args, cfg):
        self.set_my_env(args, cfg)

        # get renderer ----------------------
        if args.eval_only or cfg.TEST.SAVE_RESULTS_ONLY or (not cfg.MODEL.POSE_NET.XYZ_ONLINE):
            renderer = None
        else:
            train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            data_ref = ref.__dict__[train_dset_meta.ref_key]
            train_obj_names = train_dset_meta.objs
            render_gpu_id = self.local_rank
            renderer = get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=render_gpu_id)

        logger.info(f"Used GDRN module name: {cfg.MODEL.POSE_NET.NAME}")
        model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
        logger.info("Model:\n{}".format(model))

        # don't forget to call `setup` to prepare for model / optimizer for distributed training.
        # the model is moved automatically to the right device.
        if optimizer is not None:
            model, optimizer = self.setup(model, optimizer)
        else:
            model = self.setup(model)

        if True:
            # sum(p.numel() for p in model.parameters() if p.requires_grad)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info("{}M params".format(params))

        if cfg.TEST.SAVE_RESULTS_ONLY:  # save results only ------------------------------
            MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return self.do_save_results(cfg, model)

        if args.eval_only:  # eval only --------------------------------------------------
            MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return self.do_test(cfg, model)

        self.do_train(cfg, args, model, optimizer, renderer=renderer, resume=args.resume)
        if hard_limit < FILE_LIMIT:
            logger.warning("set sharing strategy for multiprocessing to file_system")
            torch.multiprocessing.set_sharing_strategy("file_system")
        return self.do_test(cfg, model)

class Inference(Lite):
    def __init__(self, args, cfg):
        super().__init__(
            accelerator = 'gpu',
            strategy = args.strategy,
            devices = args.num_gpus,
            num_nodes = args.num_machines,
            precision = 16 if cfg.SOLVER.AMP.ENABLED else 32,
        )
        self.model = None
        self.cfg = cfg
        self.init_model(args, cfg)
        self.setup(self.model)
        self.objects = ref.neura_object.objects
        MyCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(args.model_path, resume=False)
        logger.info(self.model)
        self.model_info = ref.__dict__["neura_object"].get_models_info()
        logger.info(self.model_info)
        MetadataCatalog.get("neura_object").set(
            ref_key="neura_object",
            objs= self.objects,
        )
    def predict(self, image, bbox, obj_id, depth = None):
        h, w, c = image.shape
        camera_matrix = np.array([[528.466,     0.,         325.0393],
                                  [0.,      530.59326,      240.08675],
                                  [0,            0,         1]])
        
        bbox_2d = np.asarray(bbox).reshape(1,4)
        data = [{
            "dataset_name": "neura_object",
            "height": h,
            "width": w,
            "cam": camera_matrix,
            "image": image,
            "img_type": "real",
            "depth": depth,
            "annotations": [{
                "category_id": obj_id,
                "bbox": bbox_2d,
                "bbox_mode": BoxMode.XYXY_ABS,
                "model_info": self.model_info[obj_id]
            }]
        }]

        dataset = GDRN_DatasetFromList(self.cfg, split="inference", lst=data, flatten=False)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=None,
            collate_fn=trivial_batch_collator,
        )
        data = next(iter(data_loader))
        with torch.no_grad():
            batch = batch_data_test(self.cfg, data)
            logger.info(batch)
            output = self.model(
                batch.get("roi_img", None),
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

        logger.info(f"Rotation output: {output['rot']}, translation output: {output['trans']}")
        return output

    def init_model(self, args, cfg):
        self.set_my_env(args, cfg)
        model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
        self.model = model

@loguru_logger.catch
def main(args):
    from pathlib import Path
    cfg = setup(args)
    # logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")
    # if args.num_gpus > 1 and args.strategy is None:
    #     args.strategy = "ddp"
    # Lite(
    #     accelerator="gpu",
    #     strategy=args.strategy,
    #     devices=args.num_gpus,
    #     num_nodes=args.num_machines,
    #     precision=16 if cfg.SOLVER.AMP.ENABLED else 32,
    # ).run(args, cfg)

    image = cv2.imread(str(Path(cur_dir).parent.parent / "data/BOP_DATASETS/neura_objects/test/rgb/scene_000000_frame_000000.jpg"))
    bbox = np.array([[374, 224,  61,  78]])
    obj_id = 2
    inference = Inference(args, cfg)
    inference.predict(image, bbox, obj_id)
    
if __name__ == "__main__":
    import resource

    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    FILE_LIMIT = 500000
    soft_limit = min(FILE_LIMIT, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    parser = my_default_argument_parser()
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="the strategy for parallel training: dp | ddp | ddp_spawn | deepspeed | ddp_sharded",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path to model weights"
    )
    args = parser.parse_args()
    iprint("Command Line Args: {}".format(args))

    if args.eval_only and hard_limit < FILE_LIMIT:
        iprint("set sharing strategy for multiprocessing to file_system")
        torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
