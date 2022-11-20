from MY_CNN import CNN
from MY_CNN import read_Dataset
demo=CNN(3,2,[28,28],3,10)#minist数据集图像大小：28*28 0到9共10种数字
data,target=read_Dataset("E:\机器学习\大作业\mnist_train.csv")
demo.train(data,target,1,0.001)