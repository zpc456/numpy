import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
MAXINT=1e10
def distance(vex1, vex2):
    # 需要自己实现，实现两个数据之间距离的计算
    distance=np.sqrt(np.sum(np.power(vex1-vex2,2)))
    return distance

def kMeans_way(S, k, distMeas=distance):
    # 数据行数
    m = np.shape(S)[0]

    sampleTag = np.zeros(m)
    # 数据列数，数据有几个属性
    n = np.shape(S)[1]
    # print (m,n)
    # 此处为初始化簇中心
    clusterCenter = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(S[:, j])
        maxJ = max(S[:, j])
        rangeJ = float(maxJ - minJ)
        clusterCenter[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))

    # print (clusterCenter)
    sampleTagChanged = True
    SSE = 0.0
    # 更新簇中心
    while sampleTagChanged:
        # 计算数据与各簇中心的距离
        # 对数据归属的簇进行改变，并记录
        sampleTagChanged=False
        for i in range(m):
            minDis=MAXINT
            minindex=-1
            for j in range(k):
                dis=distMeas(S[i,:],clusterCenter[j,:])
                if dis<minDis:
                    minDis=dis
                    minindex=j
            if minindex!=sampleTag[i]:
                sampleTag[i]=minindex
                sampleTagChanged=True
        for j in range(k):
            clusterCenter[j]=np.mean(S[sampleTag==j],axis=0)
        draw_pic(S,sampleTag,clusterCenter)
        # 计算SSE
        SSE=0.0
        for i in range(m):
            SSE=SSE+distMeas(S[i],clusterCenter[int(sampleTag[i])])
    # 需返回簇中心clusterCenter，数据类别sampleTag和SSE
    return clusterCenter, sampleTag, SSE


# 结果可视化
def draw_pic(samples, sampleTag, clusterCenter):
    k = len(clusterCenter)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    markers = ['sg', 'py', 'ob', 'pr']
    for i in range(k):
        data_pos = samples[sampleTag == i]
        plt.plot(data_pos[:, 0].tolist(), data_pos[:, 1].tolist(), markers[i])
    plt.plot(clusterCenter[:, 0].tolist(), clusterCenter[:, 1].tolist(), "r*", markersize=20)
    plt.title('鸢尾花')
    plt.show()


def tryKmeans():
    k = 3
    iris_data = load_iris()
    data = iris_data.data[:,2:]
    clusterCenter, sampleTag, SSE = kMeans_way(data, k)
    if np.isnan(clusterCenter).any():
        print("Error!reson:质心重叠！")
        print("将试第二次")
        return 0
    print(type(sampleTag))
    draw_pic(data, sampleTag, clusterCenter)
    print("----------end-------------")
    res = 1
    return 1


if __name__ == '__main__':
    while (tryKmeans() == 0):
        print("-------------ing------------")
