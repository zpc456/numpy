import numpy as np
import matplotlib.pyplot as plt


# 读入文件的地址，返回数据集的数据和标签
def read_data(path):
    print("数据加载ing...")
    data_file = open(path, 'r')
    data_list = data_file.readlines()
    data_file.close()

    # 用来存放标签和数据
    target = []
    data = []

    print('总计需加载数据个数:' + str(len(data_list)))

    # 对每行数据读入
    for j in range(len(data_list)):
        line_ = data_list[j].split(',')  # csv文件每行转列表

        numbers = [int(x) / 255 for x in line_[1:]]  # 字符串转数字列表

        numbers = np.array(numbers).reshape(28, 28)  # 转为np数组，并转换成28*28的形状

        target.append(int(line_[0]))
        data.append(numbers)

        if j % 4000 == 0:
            print('已加载 ' + str(j * 100 / len(data_list)) + '%')

    target = np.array(target)
    data = np.array(data)

    print('加载完成!')
    return data, target


# 卷积层模板
class conv:
    # 生成卷积模板
    def __init__(self, measure, num):
        """
        measure: 卷积核的尺寸
        num: 卷积核的个数
        """
        self.measure = measure
        self.num = num

        # 随机生成模板,num*measure*measure的卷积核
        self.filtres = np.random.randn(num, measure, measure) / (measure ** 2)

        # 为了保持卷积后的图像大小不变，需要在边缘增加一圈数据
        self.edge = measure // 2

    # 将原图像所感受的局部视野提取出来
    def sliding(self, image):
        """
        作为一个生成器器，返回图片中的某一局部视野，方便卷积
        """
        self.input = image
        h, w = image.shape

        # 对数据进行填充，使卷积后图形尺寸不变，填充范围为edge，如需了解更多请百度numpy.pad
        pad_image = np.pad(image, ((self.edge, self.edge), (self.edge, self.edge)), 'constant', constant_values=(0, 0))
        # 迭代生成和卷积模板相卷积的图片中的范围
        for i in range(h):
            for j in range(w):
                iter_image = pad_image[i:(i + self.measure), j:(j + self.measure)]

                # 返回局部视野和对应的坐标
                yield iter_image, i, j

    # 前向传播
    def forward(self, input_image):
        # 将输入图像保存下来，方便反馈时使用
        self.last_input = input_image
        h, w = input_image.shape

        # 输出的是分别被不同卷积核卷积后的特征图，所以大小为h*w*num
        output_image = np.zeros((h, w, self.num))

        # 卷积运算
        for iter_image, i, j in self.sliding(input_image):
            output_image[i, j] = np.sum(iter_image * self.filtres, axis=(1, 2))

        # 返回结果
        return output_image

    # 反馈修改权重参数
    def feedback(self, out, learn_rate):
        # 申请一个和卷积核相仿的数组
        filters = np.zeros(self.filtres.shape)
        for iter_image, i, j in self.sliding(self.last_input):
            for f in range(self.num):
                # 将反馈回来的卷积层权重和模板走上一遭，并对模板进行修正，因为之间的的反馈数据都包含着图像感兴趣的点
                filters[f] += out[i, j, f] * iter_image
        self.filtres -= learn_rate * filters


# 池化层结构
class pooling:

    def __init__(self, poolsize):
        # 选择池化的大小
        self.size = poolsize

    def sliding(self, image):
        """
        需要注意的是这里输入的图像是已经经过卷积的三位数组了
        """
        self.last_input = image
        h = image.shape[0] // self.size
        w = image.shape[1] // self.size

        # 大致上与卷积的相似，作用是挑选出需要池化的范围
        for i in range(h):
            for j in range(w):
                iter_image = image[(i * self.size):(i * self.size + self.size),
                             (j * self.size):(j * self.size + self.size)]
                yield iter_image, i, j

    def forward(self, input_image):
        # 输出的大小长宽就是原图像/池化大小
        output_image = np.zeros(
            (input_image.shape[0] // self.size, input_image.shape[1] // self.size, input_image.shape[2]))
        # 对多层特征图循环
        for iter_image, i, j in self.sliding(input_image):
            # 在每层特征图的范围中选出最大元素
            output_image[i, j] = np.amax(iter_image, axis=(0, 1))

        return output_image

    def feedback(self, backnodes):
        # 池化层输入数据，26x26x8，默认初始化为 0
        inputnodes = np.zeros(self.last_input.shape)

        # 每一个 iter_image 都是一个 3x3x8 的8层小矩阵
        # 修改 max 的部分，首先查找 max
        for iter_image, i, j in self.sliding(self.last_input):
            h, w, f = iter_image.shape
            # 获取 iter_image 里面最大值的索引向量，一叠的感觉
            amax = np.amax(iter_image, axis=(0, 1))

            # 遍历整个 iter_image，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # 如果这个像素是最大值，复制梯度到它。
                        if iter_image[i2, j2, f2] == amax[f2]:
                            inputnodes[i * self.size + i2, j * self.size + j2, f2] = backnodes[i, j, f2]

        return inputnodes


class softmax:
    def __init__(self, input_size, outnodes):
        # 权重文件，该层的输入节点全连接输出节点
        self.weights = np.random.randn(input_size, outnodes) / input_size
        # 输出节点偏置
        self.output = np.zeros(outnodes)

    def forward(self, input_image):
        self.last_input_shape = input_image.shape
        input_image = input_image.flatten()  # 将数据转化成一维
        self.last_input = input_image  # 将该层节点记录下来，用作反馈
        length, nodes = self.weights.shape

        # 最后的概率， totals是尺寸为outnodes的一维数组
        totals = np.dot(input_image, self.weights) + self.output
        self.last_totals = totals

        # 结论
        out = np.exp(totals)
        # 将归一化后的结果返回
        return out / np.sum(out, axis=0)

    def feedback(self, gradients, learn_rate):
        """
        gradients : 反馈回来的梯度组，目前仅是正确答案所对应的下标有正确值
        learn_rate: 学习率
        """
        # 找到正确答案所对应的那个gradient
        for i, gradient in enumerate(gradients):
            if gradient == 0:
                continue

            # 得到一群1和一个正确答案所对应的非1值
            exps = np.exp(self.last_totals)
            s = np.sum(exps)

            # 反馈的数值,具体公式见注1
            out_back = -exps[i] * exps / (s ** 2)
            out_back[i] = exps[i] * (s - exps[i]) / (s ** 2)

            # 将反馈数值和概率做乘积，得到结果权重1
            out_back = gradient * out_back

            # @ 可以理解成矩阵乘法
            # 最后的输出与结果反馈的权重做点乘，获得权重的偏置
            weight_back = self.last_input[np.newaxis].T @ out_back[np.newaxis]
            inputs_back = self.weights @ out_back

            self.weights -= learn_rate * weight_back
            self.output -= learn_rate * out_back

        # 将矩阵从 1d 转为 3d
        # 1352 to 13x13x8
        return inputs_back.reshape(self.last_input_shape)


class CNN:
    def __init__(self, convsize, poolsize, image_size, channel, classis):
        """
        convsize : 卷积核视野的大小
        poolsize : 池化范围大小
        imagesize: 图片的尺寸
        channel  : 卷积核的层数
        classis  : 分类数
        """
        # 定义一个卷积层
        self.conv3 = conv(convsize, channel)
        # 定义一个池化层
        self.pool2 = pooling(poolsize)
        # 定义一个softmax层
        self.softmax_ = softmax((image_size[0] // poolsize) * (image_size[1] // poolsize) * channel, classis)

    # 训练过程
    def train(self, images, target, wheel, learn_rate):
        """
        images    : 训练用的图片组
        target    : 训练用的答案
        wheel     : 训练的轮数
        learn_rate: 学习率
        """
        # 记录损失的函数
        loss = []
        # 计次
        item = 0
        # 绘图窗口打开
        plt.ion()
        for i in range(wheel):
            item_loss = 0  # 每轮损失函数计算
            for image in range(len(images)):
                # 数据的正向传播
                out = self.conv3.forward(images[image])
                out = self.pool2.forward(out)
                out = self.softmax_.forward(out)

                # 损失值计算
                item_loss += -np.log(out[target[image]])

                # 反馈数据
                # 仅关注正确标签，初始反馈的函数为 (-1/正确答案对应的概率)
                gradient = np.zeros(10)
                gradient[target[image]] = -1 / out[target[image]]

                gradient = self.softmax_.feedback(gradient, learn_rate)
                gradient = self.pool2.feedback(gradient)
                gradient = self.conv3.feedback(gradient, learn_rate)

                item += 1
                if item % 300 == 0:
                    plt.clf()  # 清除之前画的图
                    loss.append(item_loss / 300)
                    plt.xlabel('sample nums')
                    plt.ylabel('loss')
                    plt.title("train loss")
                    plt.plot(loss,color='red')
                    plt.pause(0.001)
                    print("process: %.4f loss: %.7f" % (item / (wheel * len(images)), item_loss / 300))
                    item_loss = 0
                    plt.ioff()

        return loss

    # 测试函数
    def test(self, image):
        # 测试函数仅包含正向传播
        out = self.conv3.forward(image)
        out = self.pool2.forward(out)
        out = self.softmax_.forward(out)

        return out, np.argmax(out)

demo = CNN(3, 2, [28, 28], 3, 10)
data, target = read_data('mnist_train.csv')
demo.train(data, target, 1, 0.001)

data2, target2 = read_data("mnist_test.csv")
count = 0
item_loss = 0
for item in range(len(data2)):
    out, result = demo.test(data2[item])
    if result == target2[item]:
        count += 1

    item_loss += -np.log(out[target2[item]])

print("准确率为：", count / (len(data2)))
print("average_loss：", item_loss / (len(data2)))