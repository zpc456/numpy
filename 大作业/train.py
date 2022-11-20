from CNN import CNN
from CNN import read_data
demo = CNN(3, 2, [28, 28], 3, 10)
data, target = read_data('mnist_train.csv')
demo.train(data, target, 1, 0.001)