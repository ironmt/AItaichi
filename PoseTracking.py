import argparse
import cv2
import mmcv
import numpy as np
import os
import csv
import sys

sys.path.append('/home/service/competition/17501024327/pyskl')
os.chdir('/home/service/competition/17501024327/pyskl')

import datetime
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer
# from pyskl.apis import *

import pickle

std_dir = '/home/service/competition/17501024327/std.pkl'

# 定义每个标签对应的关键帧
frame_key = {
    0: [90, 270, 450, 510, 720, 990],
    1: [180, 270, 450, 570, 690, 930],
    2: [150, 270, 450, 600, 780],
    3: [240, 420, 480, 570, 780, 870],
    4: [330, 450, 570, 720, 780, 900, 990, 1140],
    5: [300, 420, 480, 690, 870, 960, 1020, 1290],
    6: [210, 270, 390, 450, 510, 630, 690],
    7: [210, 270, 390, 480, 600, 720, 840, 930],
    8: [150, 240, 390, 540, 660, 750, 900, 990, 1260],
    9: [90, 210, 330, 480, 540, 750, 810, 1020, 1110],
    10: [90, 210, 360, 510, 630, 810, 960],
    11: [90, 150, 270, 360, 480, 570, 690, 780],
    12: [120, 210, 360, 600, 690, 780, 870],
    13: [120, 210, 270, 360, 420, 540, 660, 750],
}

# 定义帧率，每秒30帧
FPS = 30.0

# 定义时间窗口，向前1000ms，向后300ms
WINDOW_PRE = 1000  # 毫秒
WINDOW_POST = 300  # 毫秒

# # 定义运动检测的移动阈值
# MOVEMENT_THRESHOLD = {
#     0:0.663,
#     1:0.822,
#     2:0.733,
#     3:0.378,
#     4:0.467,
#     5:0.2,
#     6:0.467,
#     7:0.378,
#     8:0.566,
#     9:0.467,
#     10:0.25,
#     11:0.344,
#     12:1.0,
#     13:0.25
# }

# #关键帧权重
# WIGHT = {
#     0:0.233,
#     1:0.467,
#     2:0.5,
#     3:0.2,
#     4:0.122,
#     5:0.1,
#     6:0.278,
#     7:0.456,
#     8:0.1,
#     9:0.05,
#     10:0.05,
#     11:0.350,
#     12:0.550,
#     13:0.344
# }
# 定义运动检测的移动阈值
MOVEMENT_THRESHOLD = {
    0: 0.7517241379310344,
    1: 0.7827586206896552,
    2: 0.7206896551724138,
    3: 0.3793103448275862,
    4: 0.6586206896551724,
    5: 0.3793103448275862,
    6: 0.4413793103448276,
    7: 0.34827586206896555,
    8: 0.596551724137931,
    9: 1.0,
    10: 0.596551724137931,
    11: 0.6586206896551724,
    12: 1.0,
    13: 0.1
}

#关键帧权重
WIGHT = {
    0: 0.17413793103448275,
    1: 0.42241379310344823,
    2: 0.5155172413793103,
    3: 0.08103448275862069,
    4: 0.05,
    5: 0.05,
    6: 0.14310344827586208,
    7: 0.5775862068965517,
    8: 0.05,
    9: 0.05,
    10: 0.05,
    11: 0.32931034482758614,
    12: 0.5775862068965517,
    13: 0.32931034482758614
}

def calculate_angle_between_points(p1, p2, p3):
    """
    计算由三个点形成的角度（以度为单位）。
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 * norm_v2 == 0:
        return 0
    cosine_similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle_radians = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def keypoint_angle_differences(tensor1, tensor2):
    """
    计算两个关键点张量之间的八个角度差异。
    """
    # 定义每个角度的关节连接
    angle_definitions = [
        (8, 6, 12),
        (6, 8, 10),
        (7, 5, 11),
        (5, 7, 9),
        (6, 12, 14),
        (5, 11, 13),
        (12, 14, 16),
        (11, 13, 15),
    ]

    angle_differences = []

    for (joint1, joint2, joint3) in angle_definitions:
        # 从两个张量中提取三个关节的坐标
        p1 = tensor1[joint1, :2]
        p2 = tensor1[joint2, :2]
        p3 = tensor1[joint3, :2]
        p1_std = tensor2[joint1, :2]
        p2_std = tensor2[joint2, :2]
        p3_std = tensor2[joint3, :2]

        # 计算两个张量的角度
        angle_tensor1 = calculate_angle_between_points(p1, p2, p3)
        angle_tensor2 = calculate_angle_between_points(p1_std, p2_std, p3_std)

        # 计算角度差异
        angle_difference = abs(angle_tensor1 - angle_tensor2)
        angle_differences.append(angle_difference)

    return angle_differences

def calculate_one_frame_score(input_list):
    """
    根据角度差异列表计算单帧得分。
    """
    score_table = [
        {"name": "左臂弯", "weight": 0.1,  "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "左腋下", "weight": 0.25, "error_level_1": 10, "error_level_2": 30,
         "error_level_3": 50, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "右臂弯", "weight": 0.1,  "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "右腋下", "weight": 0.25, "error_level_1": 10, "error_level_2": 30,
         "error_level_3": 50, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "左腰腿", "weight": 0.1,  "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "左膝盖", "weight": 0.05, "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "右腰腿", "weight": 0.1,  "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0},
        {"name": "右膝盖", "weight": 0.05, "error_level_1": 10, "error_level_2": 20,
         "error_level_3": 30, "score_1": 1,   "score_2": 0.8, "score_3": 0.6, "score_4": 0}
    ]
    score = 0
    for i, value in enumerate(input_list):
        item = score_table[i]
        if value <= item["error_level_1"]:
            score += item["weight"] * item["score_1"]
        elif value <= item["error_level_2"]:
            score += item["weight"] * item["score_2"]
        elif value <= item["error_level_3"]:
            score += item["weight"] * item["score_3"]
        else:
            score += item["weight"] * item["score_4"]
    return score

def is_user_moving(current_skeleton, previous_skeleton,theLabel):
    """
    判断用户是否在运动，基于当前帧和上一帧的骨骼点距离。
    对 current_skeleton 和 previous_skeleton 进行归一化处理，以避免数值过大导致的溢出。
    """
    if previous_skeleton is None:
        return True
    
    # 归一化函数
    def normalize(skeleton):
        min_vals = np.min(skeleton, axis=0)
        max_vals = np.max(skeleton, axis=0)
        range_vals = max_vals - min_vals
        # 防止除以零
        range_vals[range_vals == 0] = 1
        normalized = (skeleton - min_vals) / range_vals
        return normalized
    
    # 对 current_skeleton 和 previous_skeleton 进行归一化
    current_skeleton_norm = normalize(current_skeleton.astype(np.float32))
    previous_skeleton_norm = normalize(previous_skeleton.astype(np.float32))
    
    # 计算所有关键点的欧氏距离
    distances = np.linalg.norm(current_skeleton_norm - previous_skeleton_norm, axis=1)
    
    # 计算总移动量
    total_movement = np.sum(distances)
    
    #print(f"总移动量 {total_movement:.4f}")
    return total_movement > MOVEMENT_THRESHOLD[theLabel]

# 主函数，处理视频数据并计算得分
def calculate_item(input_file , item):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    item_label = item['label']
    print(f"LABEL IS {item_label}")
    frame_dir = item['frame_dir']  # videoId

    if item_label == 14:
        return 0  # 跳过 label 为 14 的视频

    standard_item = data[item_label]

    # 获取关键帧列表
    key_frames = frame_key.get(item_label)

    # 初始化关键动作的最高得分字典
    key_action_scores = {kf: 0 for kf in key_frames}
    active_key_actions = key_frames.copy()

    # 获取权重
    weights = [WIGHT[item_label]]  # 第一个关键帧的权重
    if len(key_frames) > 1:
        weight_other = (1 - weights[0]) / (len(key_frames) - 1)
        weights.extend([weight_other] * (len(key_frames) - 1))
    else:
        weights.append(0)

    user_keypoints = item['keypoint'][0]  # 用户的关键点序列
    total_frames = item['total_frames']

    # previous_skeleton = None
    print(f"list长度 {len(user_keypoints)}")
    print(f"total_frames {total_frames}")
    # 遍历用户的视频帧
    for frame_num in range(len(user_keypoints)):
        current_time = frame_num * (1000 / FPS)  # 当前时间戳（毫秒）
        current_skeleton = user_keypoints[frame_num]
        if frame_num - 30 >= 0:
            previous_skeleton = user_keypoints[frame_num - 30]
        else :
            previous_skeleton = user_keypoints[frame_num + 30]
        # 检测用户是否在运动
        moving = is_user_moving(current_skeleton, previous_skeleton,item_label)

        if not moving:
            # print("未运动")
            continue  # 如果用户未在运动，跳过更新分数

        # 定义时间窗口
        window_start = current_time - WINDOW_PRE
        window_end = current_time + WINDOW_POST

        # 遍历关键动作
        for idx_kf, key_frame in enumerate(key_frames):
            key_time = key_frame * (1000 / FPS)  # 关键动作的时间戳（毫秒）

            if window_start <= key_time <= window_end:
                # 计算得分
                standard_frame = standard_item['keypoint'][0][key_frame]
                user_frame = user_keypoints[frame_num]
                angle_differences = keypoint_angle_differences(user_frame, standard_frame)
                score = calculate_one_frame_score(angle_differences)

                # 更新该关键动作的最高得分
                if score > key_action_scores[key_frame]:
                    key_action_scores[key_frame] = score
                    # print(f"关键动作帧 {key_frame} 得分更新为 {score:.4f}")

            # 检查是否需要输出得分
            if current_time - WINDOW_PRE > key_time and key_frame in active_key_actions:
                # 时间窗口外，输出得分
                #print(f"关键动作帧 {key_frame} 的最终得分为 {key_action_scores[key_frame]:.4f}")
                active_key_actions.remove(key_frame)

    # 计算视频的总得分
    total_score = 0
    for idx_kf, key_frame in enumerate(key_frames):
        total_score += key_action_scores[key_frame] * weights[idx_kf]
    print(f"最终得分为 {total_score:.4f}")
    return total_score



try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument(
        '--video_folder', 
        default='/home/service/video2',
        help='Folder containing video files')
    # parser.add_argument('out_filename', help='output CSV filename')
    parser.add_argument(
        '--config',
        # default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        # default='/root/AItaichi/pyskl/work_dirs/stgcn++/stgcn++_ntu60_xsub_taichi_lr4e-1_clip48_full_vidos/j/my_stgcn_ntu120_config.py',
        default='/home/service/competition/17501024327/my_posec3d_config.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        # default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        # default='/root/AItaichi/pyskl/work_dirs/stgcn++/stgcn++_ntu60_xsub_taichi_lr4e-1_clip48_full_vidos/j/epoch_159.pth',
        default='/home/service/competition/17501024327/posec3d_epoch_200.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        # default='/home/service/competition/17501024327/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        # default='/home/service/competition/17501024327/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    # parser.add_argument(
    #     '--label-map',
    #     default='tools/data/label_map/nturgbd_120.txt',
    #     # default='/root/AItaichi/pyskl/tools/data/label_map/taichi.txt',
    #     help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args

def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]

def process_video(video_path, args):
    
    # get time
    start_time = datetime.datetime.now()
    
    frame_paths, original_frames = frame_extraction(video_path,
                                                    args.short_side)
    num_frame = len(frame_paths)
    print(f"num_frame {num_frame}")
    h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    # label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
        
    kp = fake_anno['keypoint']
    if fake_anno['keypoint'] is None:
        action_label = ''
    else:
        results = inference_recognizer(model, fake_anno)
        action_label = results[0][0]
    fake_anno['keypoint'] = kp
    # print(f"\nfake_anno key lenq  {len(fake_anno['keypoint'][0])}")


    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)
    
    fake_anno['label'] = results[0][0]
    action_score = calculate_item(std_dir, fake_anno)
    
    # end time
    end_time = datetime.datetime.now()
    inference_time = end_time - start_time
    inference_time = inference_time.total_seconds() * 1000
    
    return results[0][0], action_score, inference_time
    

def process_videos_in_folder(video_folder, args):
    """Process all videos in the given folder."""
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    # label_map = [x.strip() for x in open(args.label_map).readlines()]

    data = []

    for video in videos:
        video_path = os.path.join(video_folder, video)
        print(f"Processing video: {video}")
        
        action_label, action_score, inference_time = process_video(video_path, args)

        # Assuming 'video' is the video file name without extension as unique identifier
        # video_id = os.path.splitext(video)[0]
        video_id = video

        # Append results to the data list
        data.append([video_id, action_label, action_score, inference_time])
        print(f"id: {video_id} action: {action_label} score: {action_score}")
    return data

def save_to_csv(data, out_filename):
    """Save the action recognition results to a CSV file."""
    # 确保目录存在
    directory = os.path.dirname(out_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(out_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["视频名称", "Label", "动作标准度评分", "动作分类和评分的推理总耗时"])
        writer.writerows(data)


def main():
    args = parse_args()
    
    # Process all videos in the provided folder
    data = process_videos_in_folder(args.video_folder, args)

    # 设置保存路径
    csv_dir = '/home/service/result/17501024327/17501024327_submit.csv'
    
    # 保存结果到 CSV 文件
    save_to_csv(data, csv_dir)


if __name__ == "__main__":
    main()
