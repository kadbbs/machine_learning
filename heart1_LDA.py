import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import sympy as sp


start=time.time()
heart = pd.read_csv('data/heart.csv')
print(heart)
heart.insert(0,'b',1)
print(heart)
train_data =heart[:800]
test_data =heart[800:]
print(test_data.shape)
print('test',test_data.iloc[:,14:])
#W=np.mat(np.mat(9))
W=np.mat(np.random.rand(1,14))

print(train_data)
# print(train_data.shape)

# print(train_data.iloc[:,:13])
#print(train_data.iloc[1:800,:13])
print(W)

E=np.mat('0.97466233 0.73397405 0.48049465 0.45814918 0.10962104 0.37299322 0.04923944 0.82242067 0.46382448 0.56702812 0.48822395 0.3427533 0.03206637 0.47970851')

print(train_data[0:800].shape)
# print(np.dot(W, train_data[:1].T).T)
# print(np.dot(W, train_data.T).T)



print(train_data)
#W=np.mat('129903.40753047    8535.38602082 -730196.48386484  447658.12144666 -5793.63258521    1726.31445935  -14687.10453883  179838.97775332 13842.27505103 -419251.12913644 -296031.5151313   194025.24209694 -372490.57544813 -489162.49183527')
# y=train_data.iloc[:,14:]
# print(y)
m=800
l=0.1
def hypothesis(train_data,W,m):
    return np.mat(np.dot(W,train_data.iloc[:,:14].T).T)
#print(hypothesis(train_data,W,m))

print('hyp')
def sigmoid(x):
    return 1/(1+np.exp(-x))

# def target_fun(train_data,W,m):
#     y=train_data.iloc[:,14:]

#     retuen=(-1/m)(y@math.log(hypothesis(train_data,W,m))+(1-y)@math.log(1-hypothesis(train_data,W,m)))

def grad_fun(train_data,W,m):
    x=np.mat(train_data.iloc[:,:14])
    y=np.mat(train_data.iloc[:,14:])
    h=sigmoid(hypothesis(train_data,W,m))
    print('x',x)
    print(x.shape)
    print(y.shape)
    # print(h)
    return np.dot((y-h).T,x)
print(grad_fun(train_data,W,m).shape)

# def hypothesis(train_data,W,m):
#     return np.dot(W,train_data.iloc[:,:14].T).T


def assess(test_data,W,m):
    hy=sigmoid(hypothesis(test_data,W,m))
    print('hy',hy.T)
    target=hy-test_data.iloc[:,14:]
    print('test',test_data.iloc[:,14:].T)
    not_zore=np.count_nonzero(target)

    return 1-(not_zore/(m))
if(1):
    for i in range(100000):
        #x=sp.symbols('W')
        #sp.diff(target_fun(train_data,x,m),x)
        print(i)
        W=W+l*grad_fun(train_data,W,m)
        print(W)

xn=assess(heart,W,1025)

print(xn)

end=time.time()

print('time',end-start)



