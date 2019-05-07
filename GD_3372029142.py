import re
import matplotlib.pyplot as plt
import numpy as np

X=[]
Y=[]
ox=[]
oy=[]
total=[]
with open("candyshop_data.txt",'r') as f:
    L=f.readlines();
    for a in L:
        S=re.findall("\d+\.?\d*",a)
        temp=[float(S[0]),float(S[1])]
        total.append(temp)
array=np.array(total)
star=array[np.lexsort(array[:,::-1].T)]

for a in star:
    ox.append(a[0])
    oy.append(a[1])
    X.append(a[0])
    Y.append(a[1])
X=np.array(X)
Y=np.array(Y)
beta=[1,1]
alpha=0.01
tol=0.01

def compute_grad(beta,X,Y):
    grad=[0,0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * X - Y)
    grad[1] = 2. * np.mean(X * (beta[0] + beta[1] * X - Y))
    return np.array(grad)

def update_beta(beta,alpha,grad):
    new_beta=np.array(beta)-alpha*grad
    return new_beta

def compute_cost(beta,X,Y):
    squared_err = (beta[0] + beta[1] * X - Y) ** 2
    res = np.mean(squared_err)
    return res

cost=[]
index=[]
grad=compute_grad(beta,X,Y)
i=1
index.append(i)
beta=update_beta(beta,alpha,grad)
grad=compute_grad(beta,X,Y)
cost_temp=compute_cost(beta,X,Y)
cost.append(cost_temp)
c=np.square(grad[0])
d=np.square(grad[1])
grad_abs=np.sqrt(c+d)

while(grad_abs>tol):
    i=i+1
    index.append(i)
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, X, Y)
    cost_temp = compute_cost(beta, X, Y)
    cost.append(cost_temp)
    c = np.square(grad[0])
    d = np.square(grad[1])
    grad_abs = np.sqrt(c + d)

index=np.array(index)
cost=np.array(cost)
print(beta)
plt.plot(index, cost , color='red', linewidth=2)
plt.show()











