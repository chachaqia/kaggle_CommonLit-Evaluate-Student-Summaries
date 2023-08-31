import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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

# Random Forest Regression
for i in range(1, 51):
    start_time = time.time()
    forest1 = RandomForestRegressor(n_estimators=i, random_state=42)
    forest2 = RandomForestRegressor(n_estimators=i, random_state=42)

    forest1.fit(x_train, ytrain_content)
    forest2.fit(x_train, ytrain_wording)

    pred1 = forest1.predict(x_test)
    pred2 = forest2.predict(x_test)

    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
    score = round(np.mean([rmse1, rmse2]), 3)
    end_time = time.time()
    tmp_time = round(end_time-start_time, 3)
    tmp = [i, score, tmp_time]
    print(tmp)
    losses.append(tmp)

write_txt(losses, 'random_forest_result.txt')

plt.plot(losses)
plt.xlabel('number of decision trees')
plt.ylabel('loss')
plt.title('Random Forest regression')

plt.show()
# start_time = time.time()
# forest1 = RandomForestRegressor(n_estimators=35, random_state=42)
# forest2 = RandomForestRegressor(n_estimators=35, random_state=42)
#
# forest1.fit(x_train, ytrain_content)
# forest2.fit(x_train, ytrain_wording)
#
# pred1 = forest1.predict(x_test)
# pred2 = forest2.predict(x_test)
#
# rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
# rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
# score = np.mean([rmse1, rmse2])
# end_time = time.time()
# print(score)
# print(end_time-start_time)