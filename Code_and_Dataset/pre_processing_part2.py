import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


nd = pd.read_csv('clean_data.csv')

# 将四个不同的主题的summary分类处理
p1 = nd.loc[nd['prompt_id'] == '814d6b']
p2 = nd.loc[nd['prompt_id'] == 'ebad26']
p3 = nd.loc[nd['prompt_id'] == '3b9047']
p4 = nd.loc[nd['prompt_id'] == '39c16e']

# 将整个summary训练集分为train集和test集，8/2开（按照p1 p2 p3 p4分别8/2开），并且洗牌
s = int(p1.shape[0] * 0.8)
train = p1.iloc[:s]
val = p1.iloc[s:]

s = int(p2.shape[0] * 0.8)
train = pd.concat((train, p2.iloc[:s]), axis=0)
val = pd.concat((val, p2.iloc[s:]), axis=0)

s = int(p3.shape[0] * 0.8)
train = pd.concat((train, p3.iloc[:s]), axis=0)
val = pd.concat((val, p3.iloc[s:]), axis=0)

s = int(p4.shape[0] * 0.8)
train = pd.concat((train, p4.iloc[:s]), axis=0)
val = pd.concat((val, p4.iloc[s:]), axis=0)

train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)

train = train.sample(frac=1)
val = val.sample(frac=1)

# 计算TF值用作后续训练的feature

# 提取出train集和test集的feature（
xtrain = train.text.values
xtest = val.text.values

xtrain_number = train[['text_len', 'count_stopwords', 'emoji_len', 'abbreviation_len']]
xtest_number = val[['text_len', 'count_stopwords', 'emoji_len', 'abbreviation_len']]

# 计算test集和target集每个summary的TF值，并且向量化作为拟合模型所需的feature
tfidf = TfidfVectorizer()
tfidf.fit(np.concatenate((xtrain, xtest), axis=0))

xtrain_vec = tfidf.transform(xtrain)
xtest_vec = tfidf.transform(xtest)

# 将xtrain_vec和xtest_vec转换为DataFrame
xtrain_series = pd.Series(xtrain_vec.toarray().tolist(), name='tf_idf')
xtest_series = pd.Series(xtest_vec.toarray().tolist(), name='tf_idf')

# 将ytrain_content和ytrain_wording转换为DataFrame
y_train_df = train[['content', 'wording']]
y_test_df = val[['content', 'wording']]

xtrain_array = np.array(xtrain_series.tolist())
xtest_array = np.array(xtest_series.tolist())

# 将特征数组合并为一个新的一维数组 x
x_train = np.concatenate([xtrain_array, xtrain_number], axis=1)
x_test = np.concatenate([xtest_array, xtest_number], axis=1)

x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)

# 存储为csv文件
x_train_df.to_csv('xtrain.csv', index=False)
x_test_df.to_csv('xtest.csv', index=False)
y_train_df.to_csv('ytrain.csv', index=False)
y_test_df.to_csv('ytest.csv', index=False)
