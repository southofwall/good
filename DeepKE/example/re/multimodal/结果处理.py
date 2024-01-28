
with open("实验结果.txt", "r") as file:
    # 定义一个空列表来存储数值
    values = []
    names=[]
    # 逐行读取文件内容
    for line in file:
        # 使用split()函数分割每一行内容，得到名称和数字
        name, result = line.strip().split(":")
        # 将数字转换为整数类型
        result = int(result)
        name=str(name)
        # 将数值添加到列表中
        values.append(result)
        names.append(name)
# 打印存储的数值列表
with open("实验结果zhen.txt", "r") as file:
    # 定义一个空列表来存储数值
    values2 = []
    names2=[]
    # 逐行读取文件内容
    for line in file:
        # 使用split()函数分割每一行内容，得到名称和数字
        name, result = line.strip().split(":")
        # 将数字转换为整数类型
        result = int(result)
        name=str(name)
        # 将数值添加到列表中
        values2.append(result)
        names2.append(name)

#

# 遍历两个列表，比较文本和数值

for i in range(len(names)):
    for j in range(len(names2)):
        if names[i] == names2[j]:  # 这里只以'实验1'为例，您可以根据需要修改或删除此条件
                if values[i] != values2[j]:
                    print(names[i])
                    with open("不一致1.txt", "a") as file:# 这里只以比较数值大小为例，您可以根据需要修改操作
                        file.write(f"{names[i]}:{values[i]}\n")