# 导入所需的库
import scipy.io  # 用于读取mat文件[^1^][1]
import numpy as np  # 用于处理数组数据
import matplotlib.pyplot as plt  # 用于绘制和保存图像[^2^][2]

# 读取mat文件中的数据，假设文件名为data.mat，数据的键为IQ
mat = scipy.io.loadmat("F:\PALA\PALA_data_InSilicoFlow\PALA_data_InSilicoFlow\IQ\PALA_InSilicoFlow_IQ001.mat")
data = mat["IQ"]

# 获取数据的形状，假设为(84, 143, 1000)，即84行，143列，1000个图像
rows, cols, num_images = data.shape

# 创建一个本地文件夹，用于存放图像，假设文件夹名为images
import os

# 创建一个本地文件夹来存放图像
folder = 'images'
os.makedirs(folder, exist_ok=True)

# 遍历数据中的每个图像，将其绘制并保存为png格式的文件
for i in range(num_images):
    # 获取第i个图像的数据，转换为二维数组
    # image = data[:, :, i]
    image = np.abs(data)[:, :, i]
    image = np.reshape(image, (rows, cols))

    # 将复数数据转换为模和相位
    magnitude = np.abs(image)  # 模
    phase = np.angle(image)  # 相位
    plt.axis('off')
    plt.xticks([])  # 去刻度
    # 分别绘制模和相位
    plt.imshow(image, cmap="gray")  # 绘制模

    # 保存图像到本地文件夹，文件名为image_i.png，其中i为图像的序号
    plt.savefig(f"images/{i}.png", bbox_inches='tight', pad_inches=-0.1)

    # 关闭当前的图像窗口，以便绘制下一个图像
    plt.close()
