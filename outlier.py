# 异常值检测
# 使用算法：Tukey's Test

import numpy as np
import stats as sts

list = [1,4,8,90,98,44,35,56,2,41,11,24,23,45,500, 150]
print(list)
data = np.loadtxt("C:\\Users\\ryan_\\PycharmProjects\\House-price-prediction\\process&module\\data_c.csv",delimiter=',',skiprows=1)
def outlier_detect(num,array_candidate):
    q1 = sts.quantile(array_candidate,p=0.25)
    q3 = sts.quantile(array_candidate,p=0.75)
    k1 = 1.5
    g_min_m = q1-k1*(q3-q1)
    g_max_m = q3+k1*(q3-q1)
    if num>g_max_m or num<g_min_m:
        return True
    else:
        return False
result=[]
for col in range(data.shape[1]):
    candidate=data[:,col]
    count=0
    for element in candidate:
        if outlier_detect(element,candidate) and count not in result:
            result.append(count)
            print(result)
        count+=1




