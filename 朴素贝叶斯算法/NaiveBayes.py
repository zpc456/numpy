from numpy import *
from functools import reduce

# 广告、垃圾标识
adClass = 1


def loadDataSet():
    """函数说明:加载数据集合及其对应的分类"""
    wordsList = [['周六', '公司', '一起', '聚餐', '时间'],
                 ['优惠', '返利', '打折', '优惠', '金融', '理财'],
                 ['喜欢', '机器学习', '一起', '研究', '欢迎', '贝叶斯', '算法', '公式'],
                 ['公司', '发票', '税点', '优惠', '增值税', '打折'],
                 ['北京', '今天', '雾霾', '不宜', '外出', '时间', '在家', '讨论', '学习'],
                 ['招聘', '兼职', '日薪', '保险', '返利']]
    # 1 是广告邮件, 0 是正常邮件
    classVec = [0, 1, 0, 1, 0, 1]
    return wordsList, classVec


def doc2VecList(docList):
    """函数说明:数据进行并集操作，最后返回一个词不重复的并集"""
    a = list(reduce(lambda x, y: set(x) | set(y), docList))
    print(a)
    return a


def words2Vec(vecList, inputWords):
    """函数说明:把单词转化为词向量"""
    resultVec=[0]*len(vecList)
    for word in inputWords:
        if word in vecList:
            resultVec[vecList.index(word)]+=1
        else:
            print("没有发现此单词")
    return array(resultVec)

def trainNB(trainMatrix, trainClass):
    # 类别行数
    numTrainClass = len(trainClass)
    # 列数
    numWords = len(trainMatrix[0])
    # 全部都初始化为1， 防止出现概率为0的情况出现
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 相应的单词初始化为2
    p0Words = 2.0
    p1Words = 2.0
    # 统计每个分类的词的总数
    for i in range(numTrainClass):
        if trainClass[i] == 1:
            # 数组在对应的位置上相加
            p1Num += trainMatrix[i]
            p1Words += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Words += sum(trainMatrix[i])
    # 计算每种类型里面， 每个单词出现的概率
    # 在计算过程中，由于概率的值较小，所以我们就取对数进行比较,其中ln可替换为log的任意对数底
    p0Vec = log(p0Num / p0Words)
    p1Vec = log(p1Num / p1Words)
    # 计算在类别中1出现的概率，0出现的概率可通过1-p得到
    pClass1 = sum(trainClass) / float(numTrainClass)
    return p0Vec, p1Vec, pClass1


def classifyNB(testVec, p0Vec, p1Vec, pClass1):
    """函数说明:朴素贝叶斯分类, 返回分类结果"""
    # 原本公式为p(X1|Yj)*p(X2|Yj)*...*p(Xn|Yj)*p(Yj)
    # 因为概率的值太小了，所以我们可以取ln， 根据对数特性ln(ab) = lna + lnb， 可以简化计算
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1 - pClass1)
    if p0 > p1:
        return 0
    return 1


def printClass(words, testClass):
    if testClass == adClass:
        print(words, '推测为：广告邮件')
    else:
        print(words, '推测为：正常邮件')


def tNB():
    # 加载训练数据集
    docList, classVec = loadDataSet()
    # 生成包含所有单词的list
    allWordsVec = doc2VecList(docList)
    # 构建词向量矩阵
    trainMat = list(map(lambda x: words2Vec(allWordsVec, x), docList))
    # 训练计算每个词在分类上的概率
    # 其中p0V:每个单词在“非”分类出现的概率， p1V:每个单词在“是”分类出现的概率  pClass1：类别中是1的概率
    p0V, p1V, pClass1 = trainNB(trainMat, classVec)
    # 测试数据集
    testWords = ['公司', '聚餐', '讨论', '贝叶斯']
    # 转换成单词向量，32个单词构成的数组，如果此单词在数组中，数组的项值置1
    testVec = words2Vec(allWordsVec, testWords)
    # 通过将单词向量testVec代入，根据贝叶斯公式，比较各个类别的后验概率，判断当前数据的分类情况
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    # 打印出测试结果
    printClass(testWords, testClass)
    # 同样的步骤换一例数据
    testWords = ['公司', '保险', '金融']
    testVec = words2Vec(allWordsVec, testWords)
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    printClass(testWords, testClass)


if __name__ == '__main__':
    tNB()