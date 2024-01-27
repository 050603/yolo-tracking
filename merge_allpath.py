# 导入所需的库
import math

import numpy as np
import pandas as pd

# 读取txt文件，每一行依次存放帧数、被跟踪目标的id、x1, y1 检测框的左上坐标、x2, y2: 检测框的右下坐标
# 假设文件名为data.txt，如果不是，请修改为实际的文件名
# 假设文件中没有表头，如果有，请添加header参数
# 使用with open方法读取文件，将每一行存入lines列表中
with open('runs/track/ocsort3-0.3/mot/1.mp4.txt', 'r') as file:
    lines = file.readlines()

# 定义一个空列表，存放每一行的数据
data = []

# 遍历每一行，将每一行的数据分割成列表，并添加到data列表中
for line in lines:
    data.append(line.split())

# 将data列表转换为DataFrame格式，指定列名
# df = pd.DataFrame(data, columns=['frame', 'id', 'x1', 'y1', 'x2', 'y2'])
# 初始化一个空的DataFrame
df = pd.DataFrame(columns=['frame', 'id', 'x1', 'y1', 'x2', 'y2'])

# 遍历每一行，分割数据并添加到DataFrame中
for line in lines:
    # 假设每一行的数据都是以空格分隔的
    parts = line.strip().split()
    # 只取前6个部分
    if len(parts) >= 6:
        # 将分割后的数据转换为DataFrame的一行
        row = pd.DataFrame(
            {'frame': [int(parts[0])], 'id': [int(parts[1])], 'x1': [float(parts[2])], 'y1': [float(parts[3])],
             'x2': [float(parts[4])], 'y2': [float(parts[5])]})
        # 将这一行添加到DataFrame中
        df = df.append(row, ignore_index=True)

# 将DataFrame中的数据类型转换为数值型
df = df.apply(pd.to_numeric)

# 计算每个检测框的中心坐标
df['cx'] = df['x1'] + (df['x2'] / 2)
df['cy'] = df['y1'] + (df['y2'] / 2)

# 计算每个轨迹的长度（数据点的个数）
df['length'] = df.groupby('id')['frame'].transform('count')


# 定义一个函数，计算两个轨迹之间的相似度
# 相似度的计算方法是基于轨迹位置、轨迹的方向、轨迹的速度的
# 这里使用了简单的欧氏距离、夹角余弦、速度比例作为相似度的度量
# 你可以根据你的具体需求和标准修改这个函数
def similarity(id1, id2):
    # 获取两个轨迹的数据
    df1 = df[df['id'] == id1]
    df2 = df[df['id'] == id2]
    # 计算两个轨迹的位置距离
    dist = np.sqrt((df1['cx'].mean() - df2['cx'].mean()) ** 2 + (df1['cy'].mean() - df2['cy'].mean()) ** 2)
    # 计算两个轨迹的方向余弦
    # 假设轨迹的方向是由第一个点和最后一个点的连线决定的
    # 如果不是，请修改为实际的方向计算方法
    v1 = np.array([df1['cx'].iloc[-1] - df1['cx'].iloc[0], df1['cy'].iloc[-1] - df1['cy'].iloc[0]])
    v2 = np.array([df2['cx'].iloc[-1] - df2['cx'].iloc[0], df2['cy'].iloc[-1] - df2['cy'].iloc[0]])
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 计算两个轨迹的速度比例
    # 假设轨迹的速度是由轨迹的长度除以轨迹的帧数差得到的
    # 如果不是，请修改为实际的速度计算方法
    speed1 = df1['length'].iloc[0] / (df1['frame'].max() - df1['frame'].min())
    speed2 = df2['length'].iloc[0] / (df2['frame'].max() - df2['frame'].min())
    ratio = speed1 / speed2
    # 综合三个度量，得到一个相似度的分数
    # 这里使用了一个简单的加权平均的方法，你可以根据你的具体需求和标准修改这个方法
    # 这里的权重是随意设定的，你可以根据你的具体需求和标准修改这些权重
    score = 0.5 * (1 - dist / df['cx'].max()) + 0.2 * cos + 0.3 * (1 - abs(ratio - 1))
    # score = 0.4 * (1 - dist / df['cx'].max()) + 0.2 * cos + 0.4 * (1 - abs(ratio - 1))
    return score


# 定义一个阈值，表示两个轨迹相似的最低要求
# 这个阈值是随意设定的，你可以根据你的具体需求和标准修改这个阈值
threshold = 0.9

# 定义一个字典，存放每个轨迹的合并结果
# 字典的键是轨迹的id，值是一个列表，包含合并后的轨迹的路径和轨迹合并后包含的轨迹ID
merged = {}

# 遍历每个轨迹，与其他轨迹进行相似度的比较和合并
for id1 in df['id'].unique():
    # 如果这个轨迹已经被合并过，跳过
    if id1 in merged:
        continue
    # 初始化这个轨迹的合并结果
    merged[id1] = [id1]
    # 遍历其他轨迹，与这个轨迹进行相似度的比较
    for id2 in df['id'].unique():
        # 如果是同一个轨迹，或者这个轨迹已经被合并过，跳过
        if id1 == id2 or id2 in merged:
            continue
        # 计算两个轨迹之间的相似度
        score = similarity(id1, id2)
        # 如果相似度大于阈值，将这个轨迹合并到当前轨迹
        if score > threshold:
            merged[id1].append(id2)
            # 将这个轨迹标记为已合并
            merged[id2] = True


# 定义一个函数，根据轨迹的id，获取轨迹的路径
# 路径的格式是由中心坐标组成的字符串，用逗号分隔
def get_path(id):
    # 获取轨迹的数据
    df1 = df[df['id'] == id]
    # 获取轨迹的中心坐标
    cx = df1['cx'].tolist()
    cy = df1['cy'].tolist()
    # 将坐标转换为字符串
    path = ','.join([f'({x},{y})' for x, y in zip(cx, cy)])
    return path


# 定义一个列表，存放最终的输出结果
output = []


# 定义一个函数，去除一个列表中的重复元素
def remove_duplicates(lst):
    # 使用set转换为无序不重复的集合
    # 再使用sorted排序为有序的列表
    return sorted(set(lst))


# 定义一个函数，计算两个点之间的距离
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 定义排序方法
def sort_points(points):
    # 定义一个空列表，存储重新排序后的坐标点
    new_points = []
    # 定义一个变量，存储当前绘制点
    current = points[0]
    # 将当前绘制点添加到新列表中
    new_points.append(current)
    # 定义一个变量，存储当前绘制点的索引
    index = 0
    # 定义一个变量，存储当前绘制点的距离
    dist = 0
    # 遍历剩余的点
    for i in range(1, len(points)):
        # 定义一个变量，存储最小距离
        min_dist = math.inf
        # 定义一个变量，存储最近的点
        nearest = None
        # 定义一个变量，存储最近的点的索引
        nearest_index = None
        # 遍历剩余的点，寻找离当前绘制点最近的点
        for j in range(i, len(points)):
            # 计算距离
            d = distance(current, points[j])
            # 如果距离小于最小距离，更新最小距离，最近的点和最近的点的索引
            if d < min_dist:
                min_dist = d
                nearest = points[j]
                nearest_index = j
        # 将最近的点作为下一个绘制点
        current = nearest
        # 将最近的点的索引作为下一个绘制点的索引
        index = nearest_index
        # 将最近的点的距离累加到当前绘制点的距离
        dist += min_dist
        # 将最近的点添加到新列表中
        new_points.append(current)
    # 返回新的坐标点列表
    return new_points


# 导入孤立森林算法库
from sklearn.ensemble import IsolationForest


# 定义一个函数，用于过滤异常点
def filter_outliers(points):
    # 将点的坐标转换为二维数组
    X = np.array(points)
    # 创建一个孤立森林模型，假设异常点的比例为5%
    clf = IsolationForest(contamination=0.05)
    # 训练模型
    clf.fit(X)
    # 预测每个点是否为异常点，返回一个布尔数组
    pred = clf.predict(X)
    # 定义一个空列表，存储过滤后的点
    filtered_points = []
    # 遍历每个点和预测结果
    for point, label in zip(points, pred):
        # 如果预测结果为1，表示正常点，添加到列表中
        if label == 1:
            filtered_points.append(point)
    # 返回过滤后的点列表
    return filtered_points


# 遍历每个合并结果，去除重复的坐标位置
for id, value in merged.items():
    # 如果这个轨迹已经被合并过，跳过
    if value == True:
        continue
    # 获取这个轨迹的路径
    path = get_path(id)
    # 获取这个轨迹合并后包含的轨迹ID
    ids = ','.join([str(x) for x in value])
    # 定义一个空列表，存放合并后的轨迹的坐标位置序列
    merged_path = []
    # 遍历合并后的轨迹ID，获取每个轨迹的坐标位置序列，并添加到列表中
    for id in value:
        # 获取轨迹的数据
        df1 = df[df['id'] == id]
        # 获取轨迹的中心坐标
        cx = df1['cx'].tolist()
        cy = df1['cy'].tolist()
        # 将坐标添加到列表中
        merged_path.extend(list(zip(cx, cy)))
    # 去除列表中的重复元素
    merged_path = remove_duplicates(merged_path)

    # 对列表中的坐标点进行过滤，去除异常点
    merged_path = filter_outliers(merged_path)

    # 对列表中的坐标点进行重新排序
    merged_path = sort_points(merged_path)
    # 将列表转换为字符串
    merged_path = ','.join([f'({x},{y})' for x, y in merged_path])
    # 将这个轨迹的输出结果添加到列表中
    output.append(f'{merged_path}\t{ids}')

# 将输出结果保存到本地的txt文件中
# 假设文件名为output.txt，如果不是，请修改为实际的文件名
with open('output/output_all.txt', 'w') as f:
    for line in output:
        f.write(line + '\n')

# 打印输出结果
print('输出结果如下：')
for line in output:
    print(line)
