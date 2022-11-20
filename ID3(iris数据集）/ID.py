def entropy(data_iris):
    tmp = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    # 构建空字典用来统计各个类别的数量
    for j in range(len(data_iris)):
        if data_iris[j][-1] == 'Iris-setosa':
            tmp['Iris-setosa'] += 1
        elif data_iris[j][-1] == 'Iris-versicolor':
            tmp['Iris-versicolor'] += 1
        else:
            tmp['Iris-virginica'] += 1
    sum = returnSum(tmp)
    p1 = tmp['Iris-setosa'] / sum
    p2 = tmp['Iris-versicolor'] / sum
    p3 = tmp['Iris-virginica'] / sum
    return -p1 * log(p1) - p2 * lo
    g(p2) - p3 * log(p3)


def returnSum(myDict):  # 该函数用来计算字典值之和
    sum = 0
    for i in myDict:
        sum = sum + myDict[i]
    return sum

def make_dataset():
    f = open("G:\DataSet\iris_dataset\iris.csv")
    data = f.read()
    print(data)
    data = data.split()
    data_iris=[]
    for i in range(len(data)):
        data_iris.append(data[i].split(','))
        data_iris[i][0]=float(data_iris[i][0])
        data_iris[i][1] = float(data_iris[i][1])
        data_iris[i][2] = float(data_iris[i][2])
        data_iris[i][3] = float(data_iris[i][3])
    return data_iris

def informative_attribute(data_iris,ori_entropy):
    #最优属性，最优区间
    result=[]
    for i in range(len(data_iris[0])-1):
        #对某一attribute的任意一个值分类，遍历
        best_edge=0
        best_gain=0
        for j in range(len(data_iris)):#遍历每一个值
            edge=data_iris[j][i]
            class1=[]
            class2=[]
            for t in range(len(data_iris)):
                if data_iris[t][i]>=edge:
                    class1.append(data_iris[t])
                else :
                    class2.append(data_iris[t])
            gain= ori_entropy-(len(class1)/len(data_iris))*entropy(class1)-(len(class2)/len(data_iris))*entropy(class2)
            if gain>best_gain:
                best_gain=gain
                best_edge=edge
        result.append([best_edge,best_gain])
    return result
#Print(result)显示计算结果如下
#[[5.6, 0.3862442664692114], [3.4, 0.18570201019349386],
#[3.0, 0.6365141682948128], [1.0, 0.6365141682948128]]

def informative_attribute(data_iris,ori_entropy,list):
    #全局最优属性，最优区间
    result=[]
    best_edge = 0
    best_gain = 0
    best_class1 = []
    best_class2 = []
    flag=0
    for i in list:
        #对某一attribute的任意一个值分类，遍历

        for j in range(len(data_iris)):#遍历每一个值
            edge=data_iris[j][i]
            class1=[]
            class2=[]

            for t in range(len(data_iris)):
                if data_iris[t][i]>=edge:
                    class1.append(data_iris[t])
                else :
                    class2.append(data_iris[t])
            gain= ori_entropy-(len(class1)/len(data_iris))*entropy(class1)-(len(class2)/len(data_iris))*entropy(class2)
            if gain>best_gain:
                best_gain=gain
                best_edge=edge
                best_class1=class1
                best_class2=class2
                flag=i
        #result.append([[i,best_edge,best_gain],best_class1,best_class2])

    return [flag,best_edge,best_gain],best_class1,best_class2

data_iris=make_dataset()


result1,class2_1,class2_2=informative_attribute(data_iris,entropy(data_iris),[0,1,2,3])
print(result1)
#第一个分叉点，选择属性2,分界点为3.0,
#[2, 3.0, 0.6365141682948128]

result2_1,class3_1,class3_2=informative_attribute(class2_1,entropy(class2_1),[0,1,3])
print(result2_1)
print(class3_1)
print(class3_2)
#第二行第一个分叉点，选择属性3，分界点为1.8
#[3, 1.8, 0.4783827151228094]
result2_2,class3_3,class3_4=informative_attribute(class2_2,entropy(class2_2),[0,1,3])
print(result2_2)
#第二行第二个分叉点已达到熵最小，均为setosa，不需要继续往下分
#[0, 0, 0]

result3_1,class4_1,class4_2=informative_attribute(class3_1,entropy(class3_1),[0,1])
print(result3_1)
#第三行第一个分叉点，选择属性0，分界点为6.0
#[0, 6.0, 0.042323434148154725]
print(class4_1)#均为同一种类Virginica
print(class4_2)#观察数据可知以属性1，3.0为界继续分类

result3_2,class4_3,class4_4=informative_attribute(class3_2,entropy(class3_2),[0,1])
print(result3_2)
#第三行第二个分叉点，选择属性0，分界点为7.2
#[0, 7.2, 0.04588249960217444]
print(class4_3)#均为virginica
print(class4_4)

result5,class5_1,class5_2=informative_attribute(class4_4,entropy(class4_4),[1])
print(result5)
print(class5_1)
print(class5_2)