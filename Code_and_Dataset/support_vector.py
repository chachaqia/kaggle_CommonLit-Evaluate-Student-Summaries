import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statistics
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt

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

# Support Vector regression
losses = []
xvalue = []


def write_txt(list_data, filename):
    with open(filename, 'w') as file:
        for item in list_data:
            file.write(str(item) + '\n')


def predict_loss(x, model1, model2):
    pred1 = model1.predict(x)
    pred2 = model2.predict(x)
    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
    score = np.mean([rmse1, rmse2])
    return round(score, 3)


for i in range(1, 51):
    xvalue.append(i * 1000)
    start_time = time.time()
    svr1 = LinearSVR(epsilon=0, loss='squared_epsilon_insensitive', max_iter=i * 1000, random_state=42)
    svr2 = LinearSVR(epsilon=0, loss='squared_epsilon_insensitive', max_iter=i * 1000, random_state=42)
    svr1.fit(x_train, ytrain_content)
    svr2.fit(x_train, ytrain_wording)
    loss = predict_loss(x_test, svr1, svr2)
    end_time = time.time()
    time_tmp = round(end_time - start_time, 3)
    tmp = [i * 1000, loss, time_tmp]
    print(tmp)
    losses.append(tmp)

write_txt(losses, 'support_vector_result3.txt')

x_values = [item[0] for item in losses]
y_values = [item[1] for item in losses]
plt.plot(x_values, y_values)
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('Support Vector')
plt.show()

# start_time = time.time()
# svr1 = LinearSVR(C=0.01, epsilon=0, loss='epsilon_insensitive', max_iter=2000, random_state=42)
# svr2 = LinearSVR(C=0.01, epsilon=0, loss='epsilon_insensitive', max_iter=2000, random_state=42)
# svr1.fit(x_train, ytrain_content)
# svr2.fit(x_train, ytrain_wording)
# loss = predict_loss(x_test, svr1, svr2)
# end_time = time.time()
# print(loss)
# print(end_time-start_time)
