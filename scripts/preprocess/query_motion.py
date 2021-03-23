import os
import cv2
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import affinity_batch, dtw
import matplotlib.pylab as plt
from scipy.signal import argrelextrema


MIN_NUM_FRAMES = 25


def find_region(location):
    for idx_i, item in enumerate(location):
        if item < 0:
            idx_s = idx_i
    return idx_s

def main(args, cfg):
    action_path = os.path.join('./output/vibe_output/', args.action_path)
    videos_list = os.listdir(action_path)
    videos_list = [i for i in videos_list if i != 'query' and
                   not os.path.isfile(os.path.join(action_path, i))]
    # ===> choose the query video
    query_action = args.query
    save_path = os.path.join('./output/reconstruction/', args.action_path, query_action)
    query_action = os.path.join('query', query_action)
    videos_list.insert(0, query_action)

    flag = 0
    frames_group = []
    resolution = []
    imgs = []
    seq_frame = []
    imgs_track_ids = []
    # ======> read the initial pose from VIBE prediction
    for idx, video in enumerate(videos_list):

        vibe_output_i_path = os.path.join(action_path, video, 'vibe_output.pkl')
        vibe_output = joblib.load(vibe_output_i_path)

        imgs_path = os.path.join(action_path, video, 'images')
        # ===> obtain the resolution by sample images
        sample_img = cv2.imread(os.path.join(imgs_path, '000000.png'))
        resolution.append([sample_img.shape[1]*cfg.PREPROCESS.img_size,
                           sample_img.shape[0]*cfg.PREPROCESS.img_size])

        # deal with each track ID in video_i
        for track_id in vibe_output.keys():
            joint3d = vibe_output[track_id]['joints3d']
            pose = vibe_output[track_id]['pose']
            bbox = vibe_output[track_id]['bboxes']
            betas = vibe_output[track_id]['betas']
            verts = vibe_output[track_id]['verts']
            if vibe_output[track_id]['joints2d'] is not None:
                joint2d = vibe_output[track_id]['joints2d']
            else:
                # when the tracking is done using YOLO
                joint2d = np.zeros([bbox.shape[0], 21, 3])
            resolu = np.ones([joint2d.shape[0]]) * idx
            frame_ids = vibe_output[track_id]['frame_ids']

            for frame in frame_ids:
                img_name = '{:0>6d}.png'.format(frame)
                img_path = os.path.join(imgs_path, img_name)
                imgs.append(img_path)
                seq_frame.append('{}/{}'.format(video, frame))
                imgs_track_ids.append(track_id)
            frames_group.append(joint3d.shape[0])
            if flag == 0:
                joint3d_all = joint3d
                pose_all = pose
                bbox_all = bbox
                betas_all = betas
                joint2d_all = joint2d
                resolu_all = resolu
                verts_all = verts
                flag = 1
            else:
                joint3d_all = np.concatenate((joint3d_all, joint3d), axis=0)
                pose_all = np.concatenate((pose_all, pose), axis=0)
                bbox_all = np.concatenate((bbox_all, bbox), axis=0)
                betas_all = np.concatenate((betas_all, betas), axis=0)
                joint2d_all = np.concatenate((joint2d_all, joint2d), axis=0)
                resolu_all = np.concatenate((resolu_all, resolu), axis=0)
                verts_all = np.concatenate((verts_all, verts), axis=0)
            # joint2d = vibe_output[track_id]['joints2d']
    # ===> construct the affinity matrix
    # query_affinity_matrix = affinity_batch(joint3d_all[:frames_group[0]], joint3d_all[frames_group[0]:])
    query_affinity_matrix = affinity_batch(joint3d_all[:frames_group[0]], joint3d_all)
    groups = [0]
    for i in frames_group:
        groups.append(i+groups[-1])
    groups = np.array(groups)
    # transform the joint2d from body_21 --> body_25
    joint2d_25 = np.zeros([joint2d_all.shape[0], 25, joint2d_all.shape[2]])
    joint2d_25[:, :19, :] = joint2d_all[:, :19]
    joint2d_all = joint2d_25
    # plt.figure(); plt.imshow(query_affinity_matrix); plt.show()
    # action localization
    local_max_idx_start = argrelextrema(query_affinity_matrix[0], np.greater)[0]
    local_max_value_start = query_affinity_matrix[0, local_max_idx_start]
    # require the value > threshold
    local_max_idx_start = local_max_idx_start[local_max_value_start > 0.5]
    # local_max_idx_start = local_max_idx_start[local_max_value_start > 0.6]
    local_max_idx_end = argrelextrema(query_affinity_matrix[-1], np.greater)[0]
    local_max_value_end = query_affinity_matrix[-1, local_max_idx_end]
    local_max_idx_end = local_max_idx_end[local_max_value_end > 0.6]
    # select the clips with similar frames
    variance = 0.33
    query_frames = query_affinity_matrix.shape[0]
    frames_min = query_frames - int(query_frames * variance)
    frames_max = query_frames + int(query_frames * variance)

    coarse_clips = []
    fine_clips = []
    fine_confidence = []
    if False:
        # ===> set the clip manually
        # selected_fine_clips = [[0, 219], [235, 450]]
        # import ipdb; ipdb.set_trace(context=11)
        # selected_fine_clips = [[0, 545], [548, 548 + 565]]
        selected_fine_clips = [[0, 155], [156+150, 156+300], [156+615, 156+770], [156+788+150, 156+788+300]]
    else:

        for start_i in local_max_idx_start:
            for end_j in local_max_idx_end:
                if start_i < end_j:
                    if end_j - start_i >= frames_min and end_j - start_i <= frames_max:
                        location_s = find_region(groups - start_i)
                        location_e = find_region(groups - end_j)
                        if location_s != location_e:
                            continue
                        coarse_clips.append([start_i, end_j])
                        affinity_clips = query_affinity_matrix[:, start_i:end_j]
                        refs, choose = dtw(affinity_clips)
                        # 这样选的uix里面是第一次出现这个数字的id, ret里面是数字
                        ret, uix = np.unique(refs, return_index=True)
                        choose = choose[uix]
                        confidence = affinity_clips[ret, choose]
                        # select the fine clips
                        if confidence.mean() > 1 and confidence[confidence>1].shape[0] / query_frames > 0.7:
                        # if confidence.mean() > 0.5 and confidence[confidence>0.5].shape[0] / query_frames > 0.5:
                        # if confidence.mean() > 0.6 and confidence[confidence>0.6].shape[0] / query_frames > 0.5:
                            fine_clips.append([start_i, end_j])
                            fine_confidence.append(confidence.mean())
        # NMS
        fine_clips = np.array(fine_clips)
        fine_confidence = np.array(fine_confidence)
        sort_idx = np.argsort(fine_confidence)
        fine_confidence = np.sort(fine_confidence)

        fine_clips = fine_clips[sort_idx]
        selected_fine_clips = []
        selected_clip_conf = []
        for i in range(fine_clips.shape[0]-1, 0, -1):
            item = fine_clips[i]
            item_array = np.arange(item[0], item[1])
            if len(selected_fine_clips) == 0:
                selected_fine_clips.append(item)
                selected_clip_conf.append(fine_confidence[i])
            else:
                flag = 0
                for item_j in selected_fine_clips:
                    item_j_array = np.arange(item_j[0], item_j[1])
                    common = np.intersect1d(item_array, item_j_array)
                    if common.shape[0] / min(item_array.shape[0], item_j_array.shape[0]) > 0.2:
                        flag = 1
                        break
                if flag == 0:
                    selected_fine_clips.append(item)
                    selected_clip_conf.append(fine_confidence[i])
    # save the selected clips
    selected_result = {}
    # ===> set the clip manually
    if False:
        import ipdb; ipdb.set_trace(context=11)
        # selected_fine_clips = [[0, 219], [235, 450]]
        selected_fine_clips = [[0, 545], [548, 548+565]]
    selected_clip_imgs = dict()
    selected_gt_pose3d = dict()
    for idx, clip in enumerate(tqdm(selected_fine_clips)):

        imgs_clip = imgs[clip[0]: clip[1]]
        seq_frame_clip = seq_frame[clip[0]: clip[1]]

        # ===> remove the clips that contain more than one video or one track id
        video_start = os.path.dirname(imgs_clip[0])
        video_end = os.path.dirname(imgs_clip[-1])
        track_id_start = imgs_track_ids[clip[0]]
        track_id_end = imgs_track_ids[clip[1]]
        if video_start != video_end or track_id_start != track_id_end:
            continue
        # reorganize the selected clips using img idx
        seq_name = seq_frame_clip[0].split('/')[0]
        start_img = seq_frame_clip[0].split('/')[-1]
        end_img = seq_frame_clip[-1].split('/')[-1]
        if seq_name not in selected_clip_imgs.keys():
            selected_clip_imgs[seq_name] = [[int(start_img), int(end_img)]]
        else:
            selected_clip_imgs[seq_name].append([int(start_img), int(end_img)])

        value = {'poses': pose_all[clip[0]:clip[1]],
                 'shapes': betas_all[clip[0]:clip[1]],
                 'bboxes': bbox_all[clip[0]:clip[1]],
                 'keypoints2d': joint2d_all[clip[0]:clip[1]],
                 'resolution': resolution[int(resolu_all[clip[0]])],
                 'verts': verts_all[clip[0]:clip[1]],
                 'imgs': imgs_clip}
        # ===> obtain the corresponding images

        clip_path = os.path.join(save_path, 'video_clips', 'clip_' + str(idx))
        if not os.path.exists(clip_path):
            os.makedirs(clip_path, exist_ok=True)
            for img_i in imgs_clip:
                cmd_cp = 'cp {} {}'.format(img_i, clip_path)
                os.system(cmd_cp)

        selected_result['clip_' + str(idx)] = value

        # ===> save the gt of 3D pose if existing
        if seq_name == '1_process':

            pose3d_gt = joblib.load('/mnt/data/dataset/iMocap/1028/seq_djt/track_seq_djt.pkl')
            pose3d_idx = pose3d_gt[int(start_img): int(end_img)+1]
        else:
            pose3d_idx = None
        selected_gt_pose3d['clip_' + str(idx)] = pose3d_idx
    # joblib.dump(selected_gt_pose3d, os.path.join(save_path, 'query_pose3d_gt.pkl'))
    # import ipdb; ipdb.set_trace(context=11)

    # ===> evaluate the proposals using IOU
    # load query gt
    query_gt = joblib.load(os.path.join(action_path, 'query_gt.pkl'))
    action1 = np.array([])
    query_action1_gt = dict()
    for seq in query_gt[0].keys():
        # map the seq name
        seq_gt = query_gt[0][seq]
        if seq == 'seq2_0':
            seq = '0_process'
        if seq == 'seq1_0':
            seq = '1_process'
        query_action1_gt[seq] = seq_gt['action1']

    # calculate the iou

    for seq_i in query_action1_gt.keys():
        predicted_query = np.array([])
        gt_query = np.array([])
        for j in query_action1_gt[seq_i]:
            gt_query = np.concatenate((gt_query, np.arange(j[0], j[1] + 1)))
        for j in selected_clip_imgs[seq_i]:
            predicted_query = np.concatenate((predicted_query, np.arange(j[0], j[1] + 1)))

        intersect = np.intersect1d(predicted_query, gt_query).shape[0]
        common = predicted_query.shape[0] + gt_query.shape[0] - intersect
        iou = intersect / common
        print('seq {}: iou is {}'.format(seq_i, iou))

    joblib.dump(selected_result, os.path.join(save_path, 'preprocess_vibe_openpose.pkl'))

    print('================= END =================')


if __name__ == '__main__':
    from config.default import update_cfg

    parser = argparse.ArgumentParser()

    parser.add_argument('--action_path', type=str, default='tennis/federer')
    parser.add_argument('--query', type=str, default='query_1')
    parser.add_argument('--cfg', type=str)

    args = parser.parse_args()
    cfg = update_cfg(args.cfg)

    main(args, cfg)
