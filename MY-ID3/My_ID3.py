from math import log
import pandas as pd
def createDataSet():
    # dataSet = [[0, 0, 0, 0, 'no'],         #数据集
    #         [0, 0, 0, 1, 'no'],
    #         [0, 1, 0, 1, 'yes'],
    #         [0, 1, 1, 0, 'yes'],
    #         [0, 0, 0, 0, 'no'],
    #         [1, 0, 0, 0, 'no'],
    #         [1, 0, 0, 1, 'no'],
    #         [1, 1, 1, 1, 'yes'],
    #         [1, 0, 1, 2, 'yes'],
    #         [1, 0, 1, 2, 'yes'],
    #         [2, 0, 1, 2, 'yes'],
    #         [2, 0, 1, 1, 'yes'],
    #         [2, 1, 0, 1, 'yes'],
    #         [2, 1, 0, 2, 'yes'],
    #         [2, 0, 0, 0, 'no']]
    # labels = ['不放贷', '放贷']             #分类属性

    dataSet = pd.read_csv("G:\DataSet\iris_dataset\iris.csv", skiprows=[0],

                          names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataSet=dataSet.values.tolist()
    labels=['setosa','versicolor','virginica']

    return dataSet, labels                #返回数据集和分类属性
def calcShannonEnt(dataSet):#计算香农熵的函数
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
def splitDataSet(dataSet, axis, value):
    retDataSet = []             #创建返回的数据集列表
    for featVec in dataSet:       #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet     #返回划分后的数据集
## 计算条件熵
def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet,i,value)
        prob = len(subDataSet) / float(len(dataSet))
        ce += prob * calcShannonEnt(subDataSet)
    return ce

##计算信息增益
def calcInformationGain(dataSet,baseEntropy,i):
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)
    infoGain = baseEntropy - newEntropy
    return infoGain

## 算法框架
def chooseBestFeatureToSplitByID3(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGain = calcInformationGain(dataSet,baseEntropy,i)
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
    dataSet, features = createDataSet()
    base=0.9709505944546686
    for i in range(4):
        infoGain=calcInformationGain(dataSet,1.584962500721156,i)
        print("第%d个的信息增益为：%.3f"%(i,infoGain))
    print("最优特征索引值:" + str(chooseBestFeatureToSplitByID3(dataSet)))


