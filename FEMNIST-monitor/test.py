import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D图案

a=[[1,2,3],[4,5,6],[7,8,9]]

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x_, y_ = np.meshgrid(x, y, indexing='ij')
z_ = x_ ** 2 + y_ ** 2 + x_ / 3 + y_ / 3  # 画图所要表现出来的主函数
fig = plt.figure(figsize=(10, 10), facecolor='white')  # 创建图片
sub = fig.add_subplot(111, projection='3d')  # 添加子图，
surf = sub.plot_surface(x_, y_, z_, cmap=plt.cm.brg)  # 绘制曲面,cmap=plt.cm.brg并设置颜色cmap
cb = fig.colorbar(surf, shrink=0.8, aspect=15)  # 设置颜色棒

sub.set_xlabel(r"x axis")
sub.set_ylabel(r"y axis")
sub.set_zlabel(r"z axis")
plt.show()