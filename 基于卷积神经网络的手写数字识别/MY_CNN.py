import numpy as np
import matplotlib.pyplot as plt
#读数据集，并且返回数据集的数据和标签
def read_Dataset(path):
    print("数据加载中...")
    data_file=open(path,'r')
    data_list=data_file.readlines()
    data_file.close()
    target=[]#标签
    data=[]#数据
    print("总共需要加载的数据的个数"+str(len(data_list)))
    for i in range(len(data_list)):
        line_=data_list[i].split(',')#csv文件的每一行读成列表
        numbers=[int(x)/255 for x in line_[1:]]
        numbers=np.array(numbers).reshape(28,28)
        target.append(int(line_[0]))
        data.append(numbers)
        if i%4000==0:
            print('已加载'+str(i*100/len(data_list))+'%')
    target=np.array(target)
    data=np.array(data)
    print("加载完成！")
    return data,target
#卷积层
class conv:
    def __init__(self,measure,num):
        self.measure=measure#卷积核的尺寸
        self.num=num#卷积核的个数
        self.filtres=np.random.randn(num,measure,measure)/(measure**2)
        self.edge=measure//2#填充

    #提取原图像所感受的局部视野
    def silding(self,image):
        self.input=image
        h,w=image.shape
        pad_image=np.pad(image,((self.edge,self.edge),(self.edge,self.edge)),'constant',constant_values=(0,0))
        for i in range(h):
            for j in range(w):
                iter_image=pad_image[i:(i+self.measure),j:(j+self.measure)]
                yield iter_image,i,j

    #前向传播算法
    def forward(self,input_image):
        self.last_input=input_image
        h,w=input_image.shape

        output_image=np.zeros((h,w,self.num))
        #卷积运算
        for iter_image,i,j in self.silding(input_image):
            output_image[i,j]=np.sum(iter_image*self.filtres,axis=(1,2))
            return output_image

    #反馈修改权重参数
    def feedback(self,out,learn_rete):
        filters=np.zeros(self.filtres.shape)
        for iter_image ,i,j in self.silding(self.last_input):
            for k in range(self.num):
                filters[k]+=out[i,j,k]*iter_image
        self.filtres-=learn_rete*filters

# #relu激活层
# class Relu(object):
#     def forward(self,x):
#         self.x=x
#         return np.maximum(x,0)
#     def backward(self,delta):
#         delta[self.x<0]=0
#         return delta

#池化层(最大池化)
class pooling:
    def __init__(self,poolsize):
        self.size=poolsize
    def sliding(self,image):
        self.last_input=image
        h=image.shape[0]//self.size
        w=image.shape[1]//self.size

        for i in range(h):
            for j in range(w):
                iter_image=image[(i*self.size):(i*self.size+self.size),
                           (j*self.size):(j*self.size+self.size)]
                yield iter_image,i,j
    def forward(self,input_image):
        output_image=np.zeros(
            (input_image.shape[0]//self.size,input_image.shape[1]//self.size,input_image.shape[2]))
        for iter_image, i,j in self.sliding(input_image):
            #最大池化
            output_image[i,j]=np.amax(iter_image,axis=(0,1))
        return output_image
    def feedback(self,backnodes):
        inputnodes=np.zeros(self.last_input.shape)

        for iter_image,i,j in self.sliding(self.last_input):
            h,w,f=iter_image.shape
            amax=np.amax(iter_image,axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if iter_image[i2,j2,f2]==amax[f2]:
                            inputnodes[i*self.size+i2,j*self.size+j2,f2]=backnodes[i,j,f2]
        return inputnodes

class softmax:
    def __init__(self,input_size,outnodes):
        self.weights=np.random.randn(input_size,outnodes)/input_size
        self.output=np.zeros(outnodes)
    def forward(self,input_image):
        self.last_input_shape=input_image.shape
        input_image=input_image.flatten()
        self.last_input=input_image
        length,nodes=self.weights.shape
        totals=np.dot(input_image,self.weights)+self.output
        self.last_totals=totals
        out=np.exp(totals)
        return out/np.sum(out,axis=0)
    def feedback(self,gradients,learn_rate):
        for i,gradient in enumerate(gradients):
            if gradient==0:
                continue
            exps=np.exp(self.last_totals)
            s=np.sum(exps)
            out_back=-exps[i]*exps/(s**2)
            out_back[i]=exps[i]*(s-exps[i])/(s**2)
            out_back=gradient*out_back

            weight_back=self.last_input[np.newaxis].T @ out_back[np.newaxis]#点乘
            input_back=self.weights @ out_back
            self.weights-=learn_rate*weight_back
            self.output-=learn_rate*out_back
            return input_back.reshape(self.last_input_shape)

class CNN:
    def __init__(self,convsize,poolsize,image_size,channel,classis):
        #卷积层
        self.conv3=conv(convsize,channel) #convsize->卷积核尺寸，channel->卷积核个数
        # #激活层
        # self.relu=Relu()
        #最大池化层
        self.pool2=pooling(poolsize)
        #softmax层
        self.softmax=softmax((image_size[0]//poolsize)*(image_size[1]//poolsize)*channel,classis)

    #训练
    def train(self,images,target,wheel,learn_rate):
        loss=[]
        item=0
        plt.ion()
        for i in range(wheel):
            item_loss=0
            for image in range(len(images)):
                #正向传播
                out=self.conv3.forward(images[image])
                # out=self.relu.forward(out)
                out=self.pool2.forward(out)
                out=self.softmax.forward(out)
                #计算损失值
                item_loss+=-np.log(out[target[image]])
                #反馈数据
                gradient=np.zeros(10)
                gradient[target[image]]=-1/out[target[image]]
                gradient=self.softmax.feedback(gradient,learn_rate)
                gradient=self.pool2.feedback(gradient)
                gradient=self.conv3.feedback(gradient,learn_rate)
                item+=1
                if item%200==0:
                    plt.clf()
                    loss.append(item_loss/200)

                    plt.plot(loss,color='red')
                    plt.pause(0.001)
                    print("process:%.3f loss:%.6f"%(item/(wheel*len(images)),item_loss/200))
                    item_loss=0
                    plt.ioff()
        return loss
    def test(self,image):
        out=self.conv3.forward(image)
        # out=self.relu.forward(out)
        out=self.pool2.forward(out)
        out=self.softmax.forward(out)
        return out,np.argmax(out)
