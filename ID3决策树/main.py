# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:00:16 2018

@author: aoanng
"""

from pylab import *
import treePlotter
from ID3Tree import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
##################################

# 测试决策树的构建
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
# 绘制决策树
print(myTree)
treePlotter.createPlot(myTree)