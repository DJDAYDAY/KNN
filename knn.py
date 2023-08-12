import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn.model_selection import train_test_split
#
#


def EuDistance(x1,x2):
    result=0
    if len(x1)==len(x2):
        for i in range(len(x1)-1):
            result+=pow(x1[i]-x2[i],2)
        result=math.sqrt(result)
    else:
        print("distance error")
    return result





def MinkowskiDistance(p,x1,x2):
    result=0
    if len(x1)==len(x2):
        for i in range(len(x1)-1):
            tmp=abs(x1[i]-x2[i])
            result+=pow(tmp,p)
        result=math.sqrt(result)
    else:
        print("distance error")
    return result

def KNN(k,p,dataset,test,tag,test_label):#使用数组的argsort功能并构造数组解决问题，暴力求解
    datasize=dataset.shape[0]
    #testsize=test.shape[0]
    jishuqi=0
    accnum=0
    for vector in test:
        count=0
        dis_array=np.zeros((datasize,2))
        for data in dataset:
            #labels=data.pop()
            dis_array[count][0]=MinkowskiDistance(p=p,x1=vector,x2=data)
            dis_array[count][1]=tag[count]
            count=count+1
        dis_array=dis_array[dis_array[:,0].argsort()]
        dis_array=dis_array[:k,:]
        most=np.zeros((k,2))
        most_label=None
        index=0
        for j in range(k):
            LABEL=dis_array[j][1]
            sign=True
            for j_sub in range(index):
                if LABEL==most[j_sub][1]:#看LABEK有没有和前面哪个most里LABEL是一样的
                    most[j_sub][0]+=1
                    sign=False
            if sign==True:
                most[index][0]=1
                most[index][1]=LABEL
                index+=1
        line=np.argmax(most[:,0],axis=0)#把第0列里最大的拿出来
        answer=most[line]
        '''  if answer[1]==0:
            print("the %dth test example is   Setosa"%jishuqi)
        elif answer[1]==1:
            print("the %dth test example is   Versicolor"%jishuqi)
        else:
            print("the %dth test example is   Virginica"%jishuqi)'''
        #print(int(answer[1]),end=',')
        
        if answer[1]==test_label[jishuqi]:
            accnum+=1
        jishuqi+=1
    acc=accnum/jishuqi
    accpercent=acc*100
    #print("acc rate is %f percent"%accpercent)
    return accpercent

iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.4)#划分训练集


#绘图
final=np.zeros(13)
for i in range(2,15):
    for num in range(100):
        final[i-2]+=KNN(i,1,X_train,X_test,y_train,y_test)
    final[i-2]=final[i-2]/100
    
# encoding=utf-8
import matplotlib.pyplot as plt
 
x1 = [2,3,4,5,6,7,8,9,10,11,12,13,14]
 
# 体重
y1 = final
 
# 设置画布大小
plt.figure(figsize=(16, 4))
 
# 标题
plt.title("identify best K")
 
# 数据
plt.plot(x1, y1, label='K changes', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=10)
 
# 横坐标描述
plt.xlabel('different K')
 
# 纵坐标描述
plt.ylabel('acc')
 
# 设置数字标签
for a, b in zip(x1, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 
plt.legend()
plt.show()


                


        
        



