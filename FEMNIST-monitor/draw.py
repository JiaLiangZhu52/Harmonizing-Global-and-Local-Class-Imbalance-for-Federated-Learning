import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取数据
x_data=[]
y1_data=[]
y2_data=[]
y3_data=[]
y4_data=[]
sheet=pd.read_excel(r"F:\data.xlsx")

#数据赋值
def get():
    for row in sheet.index.values:
        doc=dict()
        doc["key0"]=sheet.iloc[row,0]
        x_data.append(doc["key0"])

        doc["key1"]=sheet.iloc[row,1]
        y1_data.append(doc["key1"])

        doc["key2"] = sheet.iloc[row, 2]
        y2_data.append(doc["key2"])

        doc["key3"] = sheet.iloc[row, 3]
        y3_data.append(doc["key3"])

        doc["key4"] = sheet.iloc[row, 4]
        y4_data.append(doc["key4"])

#进行绘图
def draw():
    plt.figure(figsize=(60,40))#设置画布大小
    plt.xlabel(u'epoch',fontsize=150)#设置x轴，并设定字号大小
    plt.tick_params(labelsize=150)  # 刻度字体的大小
    plt.ylabel(u'Test accuracy', fontsize=150)  # 设置y轴，并设定字号大小
    plt.grid(True)#是否显示网格线
    #设置坐标范围
    plt.ylim((60,100))
    plt.xlim(0,50)

    #设置坐标轴刻度
    y_ticks = np.arange(60,110,10)
    x_ticks=np.arange(0,60,10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plt.plot(x_data, y1_data, color="black", linewidth=5, linestyle='-', label='Balance', marker='x',markersize=40) #  CrossEntropy Loss
    plt.plot(x_data, y2_data, color="red", linewidth=5, linestyle='-', label='1:10', marker='*',markersize=40) # marker='*', Focal Loss
    plt.plot(x_data, y3_data, color="green", linewidth=5, linestyle='-', label='1:20',marker='^',markersize=40)  # marker='h', Ratio Loss
    plt.plot(x_data, y4_data, color="blue", linewidth=5, linestyle='-', label='1:50',  marker='o',markersize=40)  #marker='h',


    plt.rcParams.update({'font.size': 90})  # 图例字体的大小
    plt.legend(loc=4)  # 图例展示位置，数字代表第几象限
    plt.savefig('sinc.pdf')
    return plt.show()  # 显示图像

get()
draw()  # 只进行绘图