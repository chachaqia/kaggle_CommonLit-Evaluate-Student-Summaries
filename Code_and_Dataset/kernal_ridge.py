import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


xtrain = pd.read_csv('xtrain.csv')
xtest = pd.read_csv('xtest.csv')
ytrain = pd.read_csv('ytrain.csv')
ytest = pd.read_csv('ytest.csv')

# 提取出train集的target值（content评分 + wording评分）
ytrain_content = ytrain.content.values
ytrain_wording = ytrain.wording.values

# 提取出来test集的target值（content评分 + wording评分）
ytest_content = ytest.content.values
ytest_wording = ytest.wording.values

# 提取出train集和test集的feature值
x_train = xtrain.values
x_test = xtest.values

# Kernel Ridge regression
start_time = time.time()
kernel_ridge1 = KernelRidge(kernel='sigmoid')
kernel_ridge2 = KernelRidge(kernel='sigmoid')

kernel_ridge1.fit(x_train, ytrain_content)
kernel_ridge2.fit(x_train, ytrain_wording)

pred1 = kernel_ridge1.predict(x_test)
pred2 = kernel_ridge2.predict(x_test)

rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
score = np.mean([rmse1, rmse2])
end_time = time.time()
print(score)
print(end_time-start_time)
