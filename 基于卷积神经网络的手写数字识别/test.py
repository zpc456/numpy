from MY_CNN import read_Dataset
from MY_CNN import CNN
import numpy as np
data2, target2 = read_Dataset("E:\机器学习\大作业\mnist_test.csv")
demo=CNN(3,2,[28,28],3,10)#minist数据集图像大小：28*28 0到9共10种数字
count = 0
item_loss = 0
for item in range(len(data2)):
    out, result = demo.test(data2[item])
    if result == target2[item]:
        count += 1

    item_loss += -np.log(out[target2[item]])

print("准确率为：", count / (len(data2)))
print("average_loss：", item_loss / (len(data2)))
