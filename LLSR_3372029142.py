import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
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
    c=[a[0]]
    d=[a[1]]
    ox.append(a[0])
    oy.append(a[1])
    X.append(c)
    Y.append(d)
X=np.array(X)
Y=np.array(Y)
lrModel = LinearRegression()
lrModel.fit(X,Y)
print(lrModel.predict([[20000],[50000]]))
alpha = lrModel.intercept_[0]
beta = lrModel.coef_[0][0]
print(alpha)
print(beta)
plt.scatter(ox, oy, color='blue')
plt.plot(X, lrModel.predict(X), color='red', linewidth=2)
plt.show()
