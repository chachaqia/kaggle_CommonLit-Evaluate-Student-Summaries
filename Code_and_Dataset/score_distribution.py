import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from matplotlib import pyplot as plt

# xtrain = pd.read_csv('xtrain.csv')
# xtest = pd.read_csv('xtest.csv')
ytrain = pd.read_csv('ytrain.csv')
ytest = pd.read_csv('ytest.csv')

y = pd.concat([ytrain, ytest], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(y.index, y['content'], s=1)
axes[0].set_title(f'content score distribution')
axes[0].set_ylabel(r'score')
axes[1].scatter(y.index, y['wording'], s=1)
axes[1].set_title(f'wording score distribution')
axes[1].set_ylabel(r'score')

plt.tight_layout()
plt.show()


def part(name):
    num = 0
    b_3 = 0
    b_2 = 0
    b_1 = 0
    b_0 = 0
    s_0 = 0
    s_1 = 0
    s_2 = 0
    for i in y[name]:
        num += 1
        if i >= 3:
            b_3 += 1
        elif i >= 2:
            b_2 += 1
        elif i >= 1:
            b_1 += 1
        elif i >= 0:
            b_0 += 1
        elif i >= -1:
            s_0 += 1
        elif i >= -2:
            s_1 += 1
        else:
            s_2 += 1
    content0 = ['score>=3', '2<=score<3', '1<=score<2', '0<=score<1', '-1<=score<0', '-2<=score<-1', 'score<-2']
    content1 = [b_3, b_2, b_1, b_0, s_0, s_1, s_2]
    content2 = ["{:.2%}".format(b_3 / num), "{:.2%}".format(b_2 / num), "{:.2%}".format(b_1 / num),
                "{:.2%}".format(b_0 / num), "{:.2%}".format(s_0 / num), "{:.2%}".format(s_1 / num),
                "{:.2%}".format(s_2 / num)]
    content = [content0, content1, content2]
    return content


cont = part('content')
word = part('wording')
print('content score:')
print(cont[0])
print('number of sample', cont[1])
print('percentage of sample', cont[2])
print('--------')
print('wording score:')
print(word[0])
print('number of sample', word[1])
print('percentage of sample', word[2])

