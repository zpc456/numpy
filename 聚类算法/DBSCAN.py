import numpy as np
import matplotlib.pyplot as plt
from sklearn import  datasets
def DIS(pointA,pointB):
    dis=np.sqrt(np.sum(np.power(pointA-pointB,2)))
    return dis
def firCluster(datasets,r,include):
    cluster=[]
    m=np.shape(datasets)[0]
    ungrouped=np.array([i for i in range (m)])
    for i in range(m):
        tempCluster=[]
        tempCluster.append(i)
        for j in range(m):
            if(DIS(datasets[i,:],datasets[j,:])<r and i!=j):
                tempCluster.append(j)
        tempCluster=np.mat(np.array(tempCluster))
        if(np.size(tempCluster))>=include :
            cluster.append(np.array(tempCluster).flatten())
    center=[]
    n=np.shape(cluster)[0]
    for k in range (n):
        center.append(cluster[k][0])
    ungrouped=np.delete(ungrouped,center)
    return cluster,center,ungrouped

def clusterGrouped(tempcluster,centers):
    m=np.shape(tempcluster)[0]
    group=[]
    position=np.ones(m)
    unvisited=[]
    unvisited.extend(centers)
    for i in range(len(position)):
        coreNeihbor=[]
        result=[]
        if position[i]:
            coreNeihbor.extend(list(tempcluster[i][:]))
            position[i]=0
            temp=coreNeihbor
            while len(coreNeihbor)>0:
                present =coreNeihbor[0]
                for j in range(len(position)):
                    if position[j]==1:
                        same=[]
                        if(present in tempcluster[j]):
                            cluster=tempcluster[j].tolist()
                            diff=[]
                            for x in cluster:
                                if x not in temp:
                                    diff.append(x)
                            temp.extend(diff)
                            position[j]=0
                del coreNeihbor[0]
                result.extend(temp)
            group.append(list(set(result)))
        i+=1
    return group

#生成非凸数据 factor表示内外圈距离比
X,Y1 = datasets.make_circles(n_samples = 1500, factor = .4, noise = .07)


#参数选择，0.1为圆半径，6为判定中心点所要求的点个数，生成分类结果
tempcluster,center,ungrouped = firCluster(X,0.1,6)
group = clusterGrouped(tempcluster,center)


#以下是分类后对数据进行进一步处理
num = len(group)
voice = list(ungrouped)
Y = []
for i in range (num):
   Y.append(X[group[i]])
flat = []
for i in range(num):
    flat.extend(group[i])
diff = [x for x in voice if x not in flat]
Y.append(X[diff])
Y = np.mat(np.array(Y))

color = ['red','blue','green','black','pink','orange']
for i in range(num):
    plt.scatter(Y[0,i][:,0],Y[0,i][:,1],c=color[i])
plt.scatter(Y[0,-1][:,0],Y[0,-1][:,1],c = 'purple')
plt.show()
