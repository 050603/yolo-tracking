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
    IQ = for_python
    # 读取mat文件中的数据，假设文件名为data.mat，数据的键为IQ
    # print("IQ:", IQ)
    mat_to_img(IQ)
    print("图像——>视频")
    path = "images"  # 输入图片路径
    video_dir = "F:/sort/yolo_tracking"  # 输出视频路径
    fps = 30  # 跟自己的需求设置帧率
    pic_to_vid(path, video_dir, fps)  # 传入函数，转化视频
    source = "F:/sort/yolo_tracking/10000.mp4"
    sys.path.append('F:/sort/yolo_tracking')
    print("begin")
    # from examples.track import yolo_track
    from examples.track import yolo_track
    print("begin2")
    output = yolo_track(source)
    print("finish")

