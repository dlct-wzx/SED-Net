
import sys
import logging
import json
import os
import open3d as o3d

from read_config import Config
config = Config('configs/config_SEDNet_normal.yml')
GPU = config.gpu

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")

# from src.my_pointnext import PointNeXt_S_Seg

import torch
from src.SEDNet import SEDNet



from src.segment_loss import EmbeddingLoss
from src.segment_utils import SIOU_matched_segments_usecd, compute_type_miou_abc
from src.segment_utils import to_one_hot, SIOU_matched_segments

from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments

# test configs
HPNet_embed = True # ========================= default True 
NORMAL_SMOOTH_W = 0.5  # =================== default 0.5
Concat_TYPE_C6 = False # ====================== default False
Concat_EDGE_C2 = False # ====================== default False
INPUT_SIZE = 10000 # =====input pc num, default 10000
my_knn = 64 # ==== default 64
use_hpnet_type_iou = False
drop_out_num = 2000 # ====== type seg rand drop  


prefix="./parsenet/" # test dataset path prefix
starts = 0  # default 0 


if_normals = config.normals
# if_normals = False

Use_MyData = True if config.dataset == "my" else False
# =============== test dataset config

if Use_MyData:
    config.num_val = config.num_test = 2700
else:
    config.num_val = config.num_test = 4163


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

userspace = ""
Loss = EmbeddingLoss(margin=1.0)

model_inst = SEDNet(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=6,
        loss_function=Loss.triplet_loss,
        mode=5 if if_normals else 0,  
        num_channels=6 if if_normals else 3,
        combine_label_prim=True,   # early fusion
        edge_module=True,  # add edge cls module
        late_fusion=True,    # ======================================
        nn_nb=my_knn  # default is 64
    )

model_inst = model_inst.cuda( )

model_inst.eval()


state_dict = torch.load(config.pretrain_model_type_path)
state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
model_inst.load_state_dict(state_dict)

test_src = 'val'

for file in os.listdir(test_src):
    if not file.endswith(".npz"):
        continue
    points = np.load(os.path.join(test_src, file))
    points = points["xyz"]
    n = points.shape[0]
    if n >= INPUT_SIZE:
        # 如果 n >= 10000，直接随机采样 10000 行（允许重复）
        indices = np.random.choice(n, INPUT_SIZE, replace=True)
        points = points[indices]
    else:
        # 如果 n < 10000，先平铺数组，再随机采样补足
        repeat_times = INPUT_SIZE // n + 1  # 计算需要平铺的次数
        stacked_arr = np.tile(points, (repeat_times, 1))  # 平铺数组
        indices = np.random.choice(stacked_arr.shape[0], INPUT_SIZE, replace=False)
        points = stacked_arr[indices]
    points = points[np.newaxis,:]
    points = torch.from_numpy(points).float().cuda()
    print(points.shape)
    with torch.no_grad():
        embedding, _, _, edges_pred = model_inst(
            points.permute(0, 2, 1), None, False
        )
    if edges_pred is not None:
        edges = torch.softmax(edges_pred, dim=1).transpose(1, 2).squeeze(0).cpu().numpy()  # [N, 2]
        red = [255, 0, 0]
        black = [0, 0, 0]

        # edge 中0 > 1则为红色，否则为黑色
        colors = [red if x[0] < x[1] else black for x in edges]
        points = points.cpu().numpy()[0,:,:]
        xyz = np.concatenate((points, colors), axis=1)
        print(xyz.shape)
        # 保存点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(xyz[:, 3:])

        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("results/" + f"{file.split('.')[0]}_edge.ply", pcd)
        # np.savetxt("results/" + f"{file.split('.')[0]}_edge.txt", edges_pred, fmt="%0.4f", delimiter=";")
