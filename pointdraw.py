# 导入matplotlib库，用于绘制图像
import matplotlib.pyplot as plt
# 设置图像的原点为左上角
plt.rcParams['image.origin'] = 'upper'

# 定义一个空列表，用于存储坐标
coordinates = []

# 打开txt文件，读取每一行
with open("runs/track/exp/mot/1.mp4.txt", "r") as f:
    for line in f:
        # 去掉行尾的换行符
        line = line.strip()
        # 用空格分隔每个元素
        elements = line.split()
        # 取出第3,4个元素，转换为整数
        x = int(elements[2])
        y = int(elements[3])
        # 将坐标添加到列表中
        coordinates.append((x, y))

# 创建一个新的图像
plt.figure()
# 反转y轴，使得正方向向下
plt.gca().invert_yaxis()
# 绘制所有坐标的散点图
plt.scatter(*zip(*coordinates), c='black', s=1)
# 设置标题和坐标轴标签
plt.title("Cumulative plot of coordinates")
plt.xlabel("X")
plt.ylabel("Y")
# 保存图像到本地
plt.savefig("plot.png")
# 显示图像
plt.show()
