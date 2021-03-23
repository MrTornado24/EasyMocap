import os
import joblib
import json
import sys
code_path = os.path.join(os.path.dirname(__file__), '..', '..', 'code')
sys.path.append(code_path)
from mytools.camera_utils import write_camera_aist

def split_dataset(input_path, split_file, output_path):
    splitFile = open(split_file, 'r')
    inputlist = sorted(os.listdir(input_path))
    while True:
        line = splitFile.readline()
        if not line:
            break
        video = line.strip().split(' ')[0]
        if not os.path.exists(f'{output_path}/{video}/videos'):
            os.system(f'mkdir {output_path}/{video}/videos')
        if video == 'gBR_sBM_cAll_d04_mBR0_ch01':
            print("Find it!")
            for input in inputlist:
                input = input.split('.')[0]
                if input.startswith('_'.join([video.split('_')[0], video.split('_')[1]])) and input.endswith('_'.join([video.split('_')[-3], video.split('_')[-2], video.split('_')[-1]])):
                    cmd = f'cp {input_path}/{input}.mp4 {output_path}/{video}/videos'
                    os.system(cmd)


def write_camera(data_path):
    if not os.path.exists(os.path.join(data_path, 'dataset')):
        cmd = 'mkdir {}/dataset'.format(data_path)
        os.system(cmd)
    mappingFile = open(f'{data_path}/cameras_new/mapping.txt', 'r')
    while True:
        line = mappingFile.readline()
        if not line:
            break
        video, camera_setting = line.strip().split(' ')
        video_path = os.path.join(data_path, 'dataset', video)
        if not os.path.exists(video_path):
            cmd = 'mkdir {}'.format(video_path)
            os.system(cmd)
        if not os.path.isfile(os.path.join(video_path, '{}.json'.format(camera_setting))):
            cmd = 'cp aist_plusplus_final/cameras_new/{}.json {}'.format(camera_setting, video_path)
            os.system(cmd)

        with open(os.path.join(video_path, f'{camera_setting}.json'), 'r') as f:
            data = json.load(f)
        write_camera_aist(data, video_path)
        os.system(f'rm {os.path.join(video_path,camera_setting)}.yaml')


if __name__ == '__main__':
    # write_camera('aist_plusplus_final')
    input_path = '/raid/ting/datasets/aist'
    output_path = '/raid/sjx/EasyMocap/aist_plusplus_final/cameras_dataset'
    split_file = '/raid/sjx/EasyMocap/aist_plusplus_final/splits/all.txt'
    split_dataset(input_path, split_file, output_path)
