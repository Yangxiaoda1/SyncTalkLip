from os import path

import numpy as np
import argparse, os, cv2
from tqdm import tqdm
import math

import sys
sys.path.append(os.getcwd().replace('preparation', ''))
import face_detection

def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")

def process_video_file(samplename, args, fa):
    vfile = '{}/{}.mp4'.format(args.video_root, samplename)
    video_stream = cv2.VideoCapture(vfile)# 打开视频文件

    frames = []
    while 1:
        still_reading, frame = video_stream.read()# 从视频中读取帧
        if not still_reading: # 如果读取到的帧不是有效的，释放视频流并退出循环
            video_stream.release()
            break
        frames.append(frame)# 将读取到的帧添加到帧列表中
    height, width, _ = frames[0].shape

    fulldir = path.join(args.bbx_root, samplename)#bbx路径
    os.makedirs(os.path.dirname(fulldir), exist_ok=True)
    if not os.path.exists(os.path.dirname(fulldir)):
        os.makedirs(os.path.dirname(fulldir))

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    bbxs = list()
    for fb in batches:# 遍历每个批次，fb:list
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):# 遍历每个批次中的检测结果，f具体对应一帧
            if f is None:# 如果检测结果无效，创建一个默认的边界框
                htmp = int((height - 96) / 2)
                wtmp = int((width - 96) / 2)
                x1, y1, x2, y2 = wtmp, htmp, wtmp + 96, htmp + 96
                qprint('没检测到','~')
                # cv2.imwrite('/home/yangxiaoda/TalkLip/TalkLip-main/output/no/{}.jpg'.format(j),fb[j])
            # htmp = int((height - 200)/2)
            # wtmp = int((width - 150)/2)
            # x1, y1, x2, y2 = wtmp, htmp, wtmp+150, htmp+200
            else:#检测成功
                x1, y1, x2, y2 = f
                # qprint('检测到了','~')
                # cv2.imwrite('/home/yangxiaoda/TalkLip/TalkLip-main/output/yes/{}.jpg'.format(j),fb[j])
            bbxs.append([x1, y1, x2, y2])
    bbxs = np.array(bbxs)
    np.save(fulldir + '.npy', bbxs)


def main(args, fa):
    print('Started processing of {}-th rank for {} on {} GPUs'.format(args.rank, args.video_root, args.gpu))

    with open(args.filelist) as f:
        lines = f.readlines()

    filelist = [line.strip().split()[0] for line in lines]

    nlength = math.ceil(len(filelist) / args.nshard)
    start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    filelist = filelist[start_id: end_id]
    print('process {}-{}'.format(start_id, end_id))

    for vfile in tqdm(filelist):
        process_video_file(vfile, args, fa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--rank", help="the rank of the current thread in the preprocessing ", default=0, type=int)
    parser.add_argument("--nshard", help="How many threads are used in the preprocessing ", default=1, type=int)
    parser.add_argument("--gpu", help="the rank of the current thread in the preprocessing ", default=1, type=int)

    args = parser.parse_args()

    if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
        raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
    							before running this script!')

    # args.rank -= 1
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(args.gpu))

    main(args, fa)
