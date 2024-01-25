# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    for frame_idx, r in enumerate(results):

        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt(source):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8_bubble.pt',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / '',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='ocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default=source,
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.2,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.1,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', default=True,
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true', default=True,
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false', default=False,
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', default=True,
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true', default=True,
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')

    opt = parser.parse_args()
    return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(opt)
def yolo_track(source):
    print("track begin")
    opt = parse_opt(source)
    print("source:",source)
    run(opt)


# 导入所需的库
import numpy as np  # 用于处理数组数据
import matplotlib.pyplot as plt  # 用于绘制和保存图像[^2^][2]
import os
import cv2
import sys


def mat_to_img(IQ):
    # 获取数据的形状，假设为(84, 143, 1000)，即84行，143列，1000个图像
    rows, cols, num_images = IQ.shape
    # 创建一个本地文件夹来存放图像
    folder = 'images'
    os.makedirs(folder, exist_ok=True)
    print("读取到IQ，正在处理为图像")
    print("num_images：", num_images)

    # 遍历数据中的每个图像，将其绘制并保存为png格式的文件
    for i in range(num_images):
        # 获取第i个图像的数据，转换为二维数组
        image = np.abs(IQ)[:, :, i]
        # image = IQ[:, :, i]
        image = np.reshape(image, (rows, cols))

        # # 将复数数据转换为模和相位
        # magnitude = np.abs(image)  # 模
        # phase = np.angle(image)  # 相位
        plt.axis('off')
        plt.xticks([])  # 去刻度
        # 分别绘制模和相位
        # plt.imshow(magnitude, cmap="gray")  # 绘制模
        plt.imshow(image, cmap="gray")  # 绘制模
        # 保存图像到本地文件夹，文件名为image_i.png，其中i为图像的序号
        plt.savefig(f"images/{i}.png", bbox_inches='tight', pad_inches=-0.1)
        print("进度：", i)
        # 关闭当前的图像窗口，以便绘制下一个图像
        plt.close()


def pic_to_vid(P, V, F):
    path = P
    video_dir = V
    fps = F
    in_img = os.listdir(path)
    # get_key是sotred函数用来比较的元素，该处用lambda表达式替代函数。
    img_key = lambda i: int(i.split('.')[0])
    img_sorted = sorted(in_img, key=img_key)
    # 需要转为视频的图片的尺寸，这里必须和图片尺寸一致
    # w,h of image
    img = cv2.imread(os.path.join(path, img_sorted[0]))
    img_size = (img.shape[1], img.shape[0])

    seq_name = "10000"  # 获取视频名称
    video_dir = os.path.join(video_dir, seq_name + '.mp4')
    # print(img_size)
    video = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_size)  # mjpg-avi

    for item in img_sorted:
        img = os.path.join(path, item)
        img = cv2.imread(img)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('全部图片已全部转化为视频。')


# 主函数
if __name__ == '__main__':
    # IQ = for_python
    # 读取mat文件中的数据，假设文件名为data.mat，数据的键为IQ
    # print("IQ:", IQ)
    # mat_to_img(IQ)
    print("图像——>视频")
    # path = "images"  # 输入图片路径
    # video_dir = "F:/sort/yolo_tracking"  # 输出视频路径
    fps = 30  # 跟自己的需求设置帧率
    # pic_to_vid(path, video_dir, fps)  # 传入函数，转化视频
    source = "F:/sort/yolo_tracking/10000.mp4"
    sys.path.append('F:/sort/yolo_tracking')
    print("begin")
    # from examples.track import yolo_track
    output = yolo_track(source)
    print("finish")
