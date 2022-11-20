from numpy import *
import re
import csv
from functools import reduce

# 广告、垃圾标识
adClass = 1


def loadDataSet():
    smss = open("./SMSSpamCollection.txt", 'r', encoding='utf-8')

    classVec = []
    docList = []
    data = csv.reader(smss, delimiter='\t')

    for line in data:
        # 读取每个垃圾邮件，并字符串转换成字符串列表

        ## 请完成此处代码的编写
        if line[0]=='ham':
            classVec.append(0)
        else:
            classVec.append(1)
        s=' '.join(line[1:])
        s=s.split( )
        docList.append(s)
    # print(docList)
    # print(classVec)
    return docList, classVec


def textParse(bigString):
    """
    函数说明:接收一个大字符串并将其解析为字符串列表
    """
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\W+', bigString)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def doc2VecList(docList):
    """函数说明:数据进行并集操作，最后返回一个词不重复的并集"""
    a = list(reduce(lambda x, y: set(x) | set(y), docList))
    return a


def words2Vec(vecList, inputWords):
    """函数说明:把单子转化为词向量"""
    # 转化成以一维数组
    resultVec = [0] * len(vecList)
    for word in inputWords:
        if word in vecList:
            # 在单词出现的位置上的计数加1
            resultVec[vecList.index(word)] += 1
        else:
            print('没有发现此单词')

    return array(resultVec)


def trainNB(trainMatrix, trainClass):
    """函数说明:计算，生成每个词对于类别上的概率"""
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


if __name__ == '__main__':
    # 加载训练数据集
    docList, classVec = loadDataSet()

    # 生成包含所有单词的list
    allWordsVec = doc2VecList(docList)
    #print(allWordsVec)

    # 构建词向量矩阵
    trainMat = list(map(lambda x: words2Vec(allWordsVec, x), docList))

    # 训练计算每个词在分类上的概率
    # 其中p0V:每个单词在“非”分类出现的概率， p1V:每个单词在“是”分类出现的概率  pClass1：类别中是1的概率
    p0V, p1V, pClass1 = trainNB(trainMat, classVec)

    text = "As a valued customer, I am pleased to advise you that following recent review of your Mob No"
    testWords = textParse(text)
    testVec = words2Vec(allWordsVec, testWords)
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    printClass(testWords, testClass)

    text = "Please don't text me anymore. I have nothing else to say."
    testWords = textParse(text)
    testVec = words2Vec(allWordsVec, testWords)
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    printClass(testWords, testClass)

