# def get_w_h():
#     # 获取图片数组的长宽
#     path = "../datasets/GroundTruth_PALA/images/train_bubbles"  # 存放图片的目录
#     filename = os.listdir(path)[0]  # 获取指定目录下的第一个文件的名称，赋值给变量 filename。、
#     filepath = os.path.join(path, filename)  # 将目录 path 和文件名 filename 组合成一个完整的文件路径。
#     # 该文件路径将被赋值给变量 filepath。这个变量可以用于后续的文件操作，比如打开文件、读取文件内容等。
#     # 打开图片并获取其尺寸
#     with Image.open(filepath) as img:  # 打开
#         width, height = img.size
#         # print("图片尺寸为：{}x{}".format(width, height))
#     # width=143 ; height=84
#     return width, height

# 读取文件并将数据存储在列表中
data = []
with open('runs/track/oc_60db_20000/20000_60db.mp4.txt', 'r') as file:
    for line in file:
        row = line.strip().split(' ')
        data.append(row)

# 按照第二列进行排序
sorted_data = sorted(data, key=lambda x: int(x[1]))

# 对每一行进行处理
for i in range(len(sorted_data)):
    # 让每一行第三列的数加上二分之一倍第五列的数
    sorted_data[i][2] = str(float(sorted_data[i][2]) + float(sorted_data[i][4]) / 2)
    # 让每一行第四列的数加上二分之一倍第六列的数
    sorted_data[i][3] = str(float(sorted_data[i][3]) + float(sorted_data[i][5]) / 2)
    # 将每一行第一列和第三列的数交换位置，第二列和第四列的数交换位置
    sorted_data[i][0], sorted_data[i][2] = sorted_data[i][2], sorted_data[i][0]
    sorted_data[i][1], sorted_data[i][3] = sorted_data[i][3], sorted_data[i][1]
    sorted_data[i][0], sorted_data[i][1] = sorted_data[i][1], sorted_data[i][0]
    sorted_data[i][1] = str(float(sorted_data[i][1]) - float(143 / 2))
    # sorted_data[i][0] = str(float(sorted_data[i][0]) + 16)

# 只保留前三列数据，写入到新文件
with open('oc_60db_2w_track.txt', 'w') as file:
    for row in sorted_data:
        file.write(' '.join(row[:4]) + '\n')


