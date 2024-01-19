# 导入matplotlib库
import matplotlib.pyplot as plt

# 打开输入文件
with open("output/output_all.txt", "r") as f:
    # 创建一个空的图像对象
    fig = plt.figure()
    # 创建一个空的坐标轴对象
    ax = fig.add_subplot(1, 1, 1)
    # 遍历输入文件的每一行
    for line in f:
        # 去掉行尾的换行符
        line = line.strip()
        # 如果行为空，跳过
        if not line:
            continue
        # 用空格分隔行，得到坐标点和轨迹id
        points, tid = line.split("	")
        # 用eval函数将字符串转换为列表
        points = eval(points)
        # 如果坐标点列表的长度小于等于1，跳过
        if len(points) <= 2:
            continue
        # 用zip函数将列表分成x和y坐标
        x, y = zip(*points)
        # 用plot函数绘制轨迹，颜色为黑色
        ax.plot(x, y, color="black")
        # 用scatter函数在轨迹的最后一个点上画一个圆点，颜色为红色，大小为20
        ax.scatter(x[-1], y[-1], color="red", s=20)
        # 用annotate函数在圆点旁边标记轨迹id，字体大小为10，水平和垂直方向都有一定的偏移量，以避免重合
        # ax.annotate(tid, (x[-1], y[-1]), fontsize=10, xytext=(5, 5), textcoords="offset points")
    # 设置坐标轴的标签
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # 设置坐标轴的方向，使得左上角为原点
    ax.invert_yaxis()
    # 保存图像为png格式
    fig.savefig("output_all.png")
