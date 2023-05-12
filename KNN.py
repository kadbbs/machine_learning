import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import sympy as sp


heart = pd.read_csv('data/heart.csv')
#print(heart)

# data=heart.iloc[:,:13]

# print(data)

train_data =heart[:800]
test_data =heart[800:]


def knn(k,train_data,Y):

    X=np.mat(train_data.iloc[:,:13]).A
    #Y=np.mat(train_data.iloc[:1,:13]).A
    print(X.shape)
    print(type(X))
    print(X-Y)

    euc=np.mat(euclidist(X,Y)).T.A
    print(euc.shape)
    lable=np.mat(train_data.iloc[:,13:]).A
    print(lable.shape)
    print(type(lable))
    print(type(euc))
    insert_lable=np.hstack((euc,lable))
    a=insert_lable
    print('type insert',type(insert_lable))
    sort=a[np.lexsort(a[:,::-1].T)]
    print(euclidist(X,Y))

    print(type(train_data))
    print(insert_lable)
    print(insert_lable.shape)

    print(sort)

    k_array=sort[:k,1:]
    print(k_array)

    one=np.count_nonzero(k_array)
    if(one>(k-one)):
        return 1
    else:
        return 0

def euclidist(A,B):

    return np.sqrt(np.sum((A-B)**2,axis=1))

# X=np.array([[1,2,3,4],
#            [1,2,3,6]])
#Y=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])



#knn(100,train_data,Y)
k=1

def kk(test_data,k):

    test_count=225
    right_count=0
    for i in range(test_count):
        test=np.mat(test_data.iloc[:,:13])
        test_lable=np.mat(test_data.iloc[:,13:])
        Y=np.mat(test[i]).A

        ph=knn(k,train_data,Y)

        if(ph==test_lable[i]):
            right_count+=1

    
    accuracy=right_count/test_count
    print(accuracy)
    return accuracy

list=[]
for i in range(799):
    i=i+1
    re=kk(test_data,i)
    list.append(re)

print(list)