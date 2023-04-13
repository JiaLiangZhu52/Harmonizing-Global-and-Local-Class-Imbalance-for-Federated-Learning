import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
num_list = [0, 0,0, 0,  0, 0, 0,64]
#num_list = [0, 14, 69, 53, 11, 70, 79, 766, 775, 638, 538, 737, 742, 807, 852, 721, 589, 823, 846, 888, 802, 611, 634, 519, 627, 494]
name_list = ['0', '1','2','3','4','5','6','20']
b = ax.bar(name_list, num_list)

plt.bar(range(len(num_list)), num_list, color='blue', tick_label=name_list)

for a, b in zip(name_list, num_list):
    ax.text(a, b , b, ha='center', va='bottom',fontsize=13)
#plt.figure(figsize=(20,10))
plt.tick_params(labelsize=20)
# plt.xlabel('client 37',fontsize=20)
plt.title('client 37', fontsize=28)

plt.ylabel('sample quantity',fontsize=20)
plt.savefig('37.pdf')
