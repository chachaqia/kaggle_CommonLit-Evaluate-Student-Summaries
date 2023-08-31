import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

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

# 固定随机数
np.random.seed(42)
tf.random.set_seed(42)

# Neural network
start_time1 = time.time()
neural_network1 = Sequential()
neural_network1.add(Dense(20, input_dim=x_train.shape[1], activation='tanh'))  # 第一个隐藏层
neural_network1.add(Dense(1, activation='linear'))

neural_network2 = Sequential()
neural_network2.add(Dense(20, input_dim=x_train.shape[1], activation='tanh'))  # 第一个隐藏层
neural_network2.add(Dense(1, activation='linear'))

# Learning algorithm and learning rate
neural_network1.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
neural_network2.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
end_time1 = time.time()
tmp_time1 = round(end_time1 - start_time1, 3)
tmp_time2 = 0

# Training the neural network
for i in range(7):
    start_time2 = time.time()
    neural_network1.fit(x_train, np.array(ytrain_content), batch_size=500, epochs=1)
    neural_network2.fit(x_train, np.array(ytrain_wording), batch_size=500, epochs=1)

    pred1 = neural_network1.predict(x_test)
    pred2 = neural_network2.predict(x_test)
    rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
    rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5

    score = round(np.mean([rmse1, rmse2]), 3)
    end_time2 = time.time()
    tmp_time2 += round(end_time2 - start_time2, 3)

    tmp = [i, score, tmp_time1 + tmp_time2]
    losses.append(tmp)
    print(tmp)


write_txt(losses, 'neural_network_result.txt')

x_values = [item[0] for item in losses]
y_values = [item[1] for item in losses]
plt.plot(x_values, y_values)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('neural network')

plt.show()

# neural_network1.fit(x_train, np.array(ytrain_content), batch_size=500, epochs=6, validation_split=0.2)
# neural_network2.fit(x_train, np.array(ytrain_wording), batch_size=500, epochs=6, validation_split=0.2)
#
# pred1 = neural_network1.predict(x_test)
# pred2 = neural_network2.predict(x_test)
#
# rmse1 = mean_squared_error(ytest_content, pred1) ** 0.5
# rmse2 = mean_squared_error(ytest_wording, pred2) ** 0.5
#
# score = np.mean([rmse1, rmse2])
# end_time = time.time()
#
# print(score)
# print(end_time-start_time1)
