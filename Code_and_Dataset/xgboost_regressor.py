import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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

# XGBoost regression
start_time0 = time.time()
xgb1 = XGBRegressor(n_estimators=1, max_depth=20, eta=0.1, subsample=0.7, colsample_bytree=0.8, gamma=1)
xgb2 = XGBRegressor(n_estimators=1, max_depth=20, eta=0.1, subsample=0.7, colsample_bytree=0.8, gamma=1)
end_time0 = time.time()
tmp_time0 = round(end_time0-start_time0, 3)

start_time1 = time.time()
xgb1.fit(x_train, ytrain_content)
xgb2.fit(x_train, ytrain_wording)

end_time1 = time.time()
tmp_time1 = round(end_time1-start_time1, 3)

for i in range(1, 101):
    start_time2 = time.time()

    pred1 = xgb1.predict(x_test)
    pred2 = xgb2.predict(x_test)
    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
    score = round(np.mean([rmse1, rmse2]),3)

    end_time2 = time.time()
    tmp_time2 = round(end_time2-start_time2, 3)
    tmp = [i, score, tmp_time0+tmp_time1+tmp_time2]
    print(tmp)
    losses.append(tmp)

    start_time1 = time.time()
    xgb1.fit(x_train, ytrain_content, xgb_model=xgb1)
    xgb2.fit(x_train, ytrain_wording, xgb_model=xgb2)
    end_time1 = time.time()
    tmp_time1 += round(end_time1-start_time1, 3)

write_txt(losses, 'xgboost_result.txt')

x_values = [item[0] for item in losses]
y_values = [item[1] for item in losses]
plt.plot(x_values, y_values)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('xgboost regression')

plt.show()

# start_time = time.time()
# xgb1 = XGBRegressor(n_estimators=60, max_depth=20, eta=0.1, subsample=0.7, colsample_bytree=0.8, gamma=1)
# xgb2 = XGBRegressor(n_estimators=60, max_depth=20, eta=0.1, subsample=0.7, colsample_bytree=0.8, gamma=1)
#
# xgb1.fit(x_train, ytrain_content)
# xgb2.fit(x_train, ytrain_wording)
# pred1 = xgb1.predict(x_test)
# pred2 = xgb2.predict(x_test)
# rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
# rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
# score = np.mean([rmse1, rmse2])
# end_time = time.time()
# print(score)
# print(end_time-start_time)