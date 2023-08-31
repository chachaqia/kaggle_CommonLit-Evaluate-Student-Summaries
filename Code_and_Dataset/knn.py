import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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


def write_txt(list_data, filename):
    with open(filename, 'w') as file:
        for item in list_data:
            file.write(str(item) + '\n')


losses = []

# KNN regression
for i in range(200, 250):
    # weights use uniform
    knn1 = KNeighborsRegressor(n_neighbors=i, weights='uniform')
    knn2 = KNeighborsRegressor(n_neighbors=i, weights='uniform')

    knn1.fit(x_train, ytrain_content)
    knn2.fit(x_train, ytrain_wording)
    pred1 = knn1.predict(x_test)
    pred2 = knn2.predict(x_test)
    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
    score1 = round(np.mean([rmse1, rmse2]), 3)

    # weights use distance
    start_time = time.time()
    knn1 = KNeighborsRegressor(n_neighbors=i, weights='distance')
    knn2 = KNeighborsRegressor(n_neighbors=i, weights='distance')

    knn1.fit(x_train, ytrain_content)
    knn2.fit(x_train, ytrain_wording)
    pred1 = knn1.predict(x_test)
    pred2 = knn2.predict(x_test)
    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
    score2 = round(np.mean([rmse1, rmse2]), 3)
    end_time = time.time()
    tmp_time = round(end_time - start_time, 3)

    tmp = [i, score1, score2, tmp_time]
    losses.append(tmp)
    print(tmp)


write_txt(losses, 'knn_result2.txt')

x_values = [item[0] for item in losses]
y_values = [item[2] for item in losses]
plt.plot(x_values, y_values)
plt.xlabel('number of neighbors')
plt.ylabel('loss')
plt.title('KNN regression')

plt.show()

# knn1 = KNeighborsRegressor(n_neighbors=30, weights='uniform')
# knn2 = KNeighborsRegressor(n_neighbors=30, weights='uniform')
#
# knn1.fit(x_train, ytrain_content)
# knn2.fit(x_train, ytrain_wording)
#
# pred1 = knn1.predict(x_test)
# pred2 = knn2.predict(x_test)
#
# rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
# rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
# score = np.mean([rmse1, rmse2])
# losses.append(score)
# print(score)
# start_time = time.time()
# knn1 = KNeighborsRegressor(n_neighbors=30, weights='distance')
# knn2 = KNeighborsRegressor(n_neighbors=30, weights='distance')
#
# knn1.fit(x_train, ytrain_content)
# knn2.fit(x_train, ytrain_wording)
#
# pred1 = knn1.predict(x_test)
# pred2 = knn2.predict(x_test)
#
# rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
# rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
# score = np.mean([rmse1, rmse2])
# end_time = time.time()
# losses.append(score)
# print(score)
# print(end_time-start_time)