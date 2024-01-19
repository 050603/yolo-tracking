from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from pathlib import Path
from boxmot import OCSORT

tracker = OCSORT(
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='examples/weights/yolov8_bubble.pt',
    confidence_threshold=0.01,
    device="0",  # or 'cuda:0'
)

# 打开视频源，可以是摄像头或本地视频文件
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture("examples/test.mp4")

# 设置帧率
fps = 25
# 获取窗口大小
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 创建视频写入对象，指定输出文件名、编码格式、帧率和分辨率
videoWrite = cv2.VideoWriter('MySaveVideo.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

while True:
    # 读取某一帧
    ret, im = capture.read()
    # 如果没有读到帧，退出循环
    if not ret:
        break

    # get sliced predictions
    result = get_sliced_prediction(
        im,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    num_predictions = len(result.object_prediction_list)
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(result.object_prediction_list):
        dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        dets[ind, 4] = object_prediction.score.value
        dets[ind, 5] = object_prediction.category.id

    tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind)

    if tracks.shape[0] != 0:

        xyxys = tracks[:, 0:4].astype('int') # float64 to int
        ids = tracks[:, 4].astype('int') # float64 to int
        confs = tracks[:, 5].round(decimals=2)
        clss = tracks[:, 6].astype('int') # float64 to int
        inds = tracks[:, 7].astype('int') # float64 to int

        # print bboxes with their associated id, cls and conf
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                # f'id: {id}, conf: {conf}, c: {cls}',
                f'id: {id}',
                (xyxy[0], xyxy[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # 将检测后的每一帧写入输出文件
    videoWrite.write(im)
    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
capture.release()
videoWrite.release()
cv2.destroyAllWindows()
