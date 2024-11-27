import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import os.path as osp
import shutil
import torch
import warnings
import pickle
import decord
import mmcv
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from pyskl.apis import inference_recognizer, init_recognizer
import torch.multiprocessing as mp
import threading


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
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1

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

def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo - Distributed Version')
    parser.add_argument('list_file', help='text file containing video paths and labels')
    parser.add_argument('out_folder', help='output folder for pkl files')
    parser.add_argument('--config', default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py', help='skeleton action recognition config file path')
    parser.add_argument('--checkpoint', default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth', help='skeleton action recognition checkpoint file/url')
    # parser.add_argument('--checkpoint', default='/root/AItaichi/train/train_all/best_top1_acc_epoch_65.pth', help='skeleton action recognition checkpoint file/url')
    
    parser.add_argument('--det-config', default='demo/faster_rcnn_r50_fpn_1x_coco-person.py', help='human detection config file path (from mmdet)')
    parser.add_argument('--det-checkpoint', default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth', help='human detection checkpoint file/url')
    parser.add_argument('--pose-config', default='demo/hrnet_w32_coco_256x192.py', help='human pose estimation config file path (from mmpose)')
    parser.add_argument('--pose-checkpoint', default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth', help='human pose estimation checkpoint file/url')
    parser.add_argument('--device', type=str, default='cuda', help='CPU/CUDA device option')
    parser.add_argument('--short-side', type=int, default=480, help='specify the short-side length of the image')
    parser.add_argument('--det-score-thr', type=float, default=0.9, help='the threshold of human detection score')
    args = parser.parse_args()
    return args

def process_video(video_info, args, gpu_id, model_index):
    video_path, label = video_info
    label = int(label)
    device = f'cuda:{gpu_id}'
    
    frame_paths, original_frames = frame_extraction(video_path, args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results
    det_results = detection_inference(args, frame_paths, device, model_index)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results, device, model_index)
    torch.cuda.empty_cache()

    # Extract video metadata and visualize poses on frames
    model = init_pose_model(args.pose_config, args.pose_checkpoint, device)
    vid_name = osp.basename(video_path).replace('.mp4', '')
    vis_frames = [
        vis_pose_result(model, frame, pose_results[i], dataset='TopDownCocoDataset', kpt_score_thr=0.3)
        for i, frame in enumerate(original_frames)
    ]
    for frame in vis_frames:
        cv2.putText(frame, str(label), (10, 30), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

    # Save the visualized frames as a video
    video_out_path = osp.join(args.out_folder, 'videos', f'{vid_name}_vis.mp4')
    os.makedirs(osp.dirname(video_out_path), exist_ok=True)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(video_out_path, remove_temp=True)

    # Pose tracking and annotation creation
    fake_anno = dict(
        frame_dir=vid_name,
        label=label,
        img_shape=(h, w),
        original_shape=(h, w),
        total_frames=num_frame,
        num_person_raw=len(pose_results[0]) if pose_results else 0,
        keypoint=None
    )

    num_person = max([len(x) for x in pose_results])
    num_keypoint = 17  # Assuming COCO-keypoints (17 keypoints)
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    # Save annotation to pkl file
    out_pkl_path = osp.join(args.out_folder, f'{vid_name}_annotations.pkl')
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(fake_anno, f)
    

def frame_extraction(video_path, short_side):
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
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
  

def detection_inference(args, frame_paths, device, model_index):
    model = init_detector(args.det_config, args.det_checkpoint, device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print(f'\nPerforming Human Detection for each frame using model {model_index}\n')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results

def pose_inference(args, frame_paths, det_results, device, model_index):
    model = init_pose_model(args.pose_config, args.pose_checkpoint, device)
    ret = []
    print(f'\nPerforming Pose Estimation using model {model_index}\n')
    for f, d in zip(frame_paths, det_results):
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
    return ret

def main():
    args = parse_args()
    with open(args.list_file, 'r') as f:
        video_list = [line.strip().split() for line in f.readlines()]
    
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 4, 'Requires at least 4 GPUs'

    # Split videos across available GPUs
    video_chunks = [video_list[i::num_gpus] for i in range(num_gpus)]

    mp.spawn(worker, args=(video_chunks, args), nprocs=num_gpus, join=True)

def worker(gpu_id, video_chunks, args):
    torch.cuda.set_device(gpu_id)
    threads = []
    # Launch 8 models per GPU
    for model_index in range(1):
        thread = threading.Thread(target=process_videos_for_model, args=(gpu_id, video_chunks[gpu_id], args, model_index))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def process_videos_for_model(gpu_id, video_chunk, args, model_index):
    for video_info in video_chunk:
        process_video(video_info, args, gpu_id, model_index)

if __name__ == '__main__':
    main()
