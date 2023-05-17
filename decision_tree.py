import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import sympy as sp


heart = pd.read_csv('heart.csv')

train_data =np.mat(heart[:800]).A
test_data =np.mat(heart[800:]).A


train_label=train_data[:,13:]
test_label=test_data[:,13:]


gini=1-(np.count_nonzero(train_label)/800)**2-(1-(np.count_nonzero(train_label)/800))**2



def gini_fun(row,col,data):

    data[row][col]
    pass

print(gini)

print(train_data[0][13])

print(train_data)
print(train_data.shape)


print(heart)
list_1=[]
list_2=[]
for i in range(800):
    train_data[i][0]

    if(train_data[i][-1]==1):
        list_1.append(train_data[i][0])
    else:
        list_2.append(train_data[i][0])


average_1=np.mean(list_1)
average_2=np.mean(list_2)

print(average_1)
print(average_2)

print(np.median(list_1))
print(np.median(list_2))


def ligi(list):
    a=np.count_nonzero(list)/len(list)
    return 1-(a)**2-(1-a)**2
def gi(list_1, list_2):
    
    return ligi(list_1)+ligi(list_2)
# def test_(data):
list=[]
for u in range(77-29+1):

    for i in range(800):
    
        if(train_data[i][0]>=29+u):
            list_1.append(train_data[i][-1])
        else:
            list_2.append(train_data[i][-1])

    list.append(gi(list_1, list_2))

print(list)


age=train_data[:,:1].T



print(np.sort(age))