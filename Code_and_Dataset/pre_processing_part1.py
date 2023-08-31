import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# train data
summary_train = pd.read_csv('summaries_train.csv')
prompts_train = pd.read_csv('prompts_train.csv')


# function for pre-processing
def clean_first(text):
    text = str(text).lower()
    text = re.sub(r'\\', '', text)  # Escape characters
    text = re.sub(r'<.*?>', '', text)  # HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'@\S+', '', text)  # Mentions
    text = re.sub(r'#\S+', '', text)  # hashtagsfirst
    text = re.sub(r'&\S+', '', text)  # html characters
    return text


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def emoticons_to_text(text):
    EMOTICONS = {
        u"xD": "Funny face", u"XD": "Funny face",
        u":3": "Happy face", u":o": "Happy face",
        u"=D": "Laughing",
        u"D:": "Sadness", u"D;": "Great dismay", u"D=": "Great dismay",
        u":O": "Surprise", u":‑O": "Surprise", u":‑o": "Surprise", u":o": "Surprise", u"o_O": "Surprise",
        u":-0": "Shock", u":X": "Kiss", u";D": "Wink or smirk",
        u":p": "cheeky, playful", u":b": "cheeky, playful", u"d:": "cheeky, playful",
        u"=p": "cheeky, playful", u"=P": "cheeky, playful",
        u":L": "annoyed", u":S": "annoyed", u":@": "annoyed",
        u":$": "blushing", u":x": "Sealed lips",
        u"^.^": "Laugh", u"^_^": "Laugh",
        u"T_T": "Sad", u";_;": "Sad", u";n;": "Sad", u";;": "Sad", u"QQ": "Sad"
    }
    for emot in EMOTICONS:
        text = re.sub(emot, EMOTICONS[emot], text)
    return text


def abbreviation(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)

    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text


def stem(text):
    ps = PorterStemmer()
    words = ' '.join([ps.stem(word) for word in text.split() if word not in stopwords.words('english')])
    return words


def clean_last(text):
    text = re.sub(r'[^\x00-\x7f]', '', text)  # non-ASCII
    text = re.sub(r'[^a-zA-Z]', '   ', text)  # remove everything except letters
    text = re.sub(r'\s+', ' ', text)
    return text


# clean text fields
summary_train['text'] = summary_train['text'].apply(clean_first)
prompts_train['prompt_title'] = prompts_train['prompt_title'].apply(clean_first)

data = pd.merge(summary_train, prompts_train, how='left')
data.drop(columns=['prompt_text', 'student_id'], inplace=True)

# 计算出每个summary字数长度
data['text_len'] = data['text'].apply(lambda x: len(x.split(' ')))

# 计算出每个summary使用的stopword的数量
stop_words = list(stopwords.words('english'))
texts = data['text'].values
count_stopwords = []
for text in texts:
    words = text.split()
    count = 0
    for word in words:
        count += (word in stop_words)
    count_stopwords.append(count)
data['count_stopwords'] = count_stopwords

# 处理emoji
data['text'] = data['text'].apply(remove_emoji)
data['text'] = data['text'].apply(emoticons_to_text)
data['clean_emoji'] = data['text'].apply(lambda x: len(x.split(' ')))
data['emoji_len'] = data['text_len'] - data['clean_emoji']

# 处理缩写
data['text'] = data['text'].apply(abbreviation)
data['clean_abbreviation'] = data['text'].apply(lambda x: len(x.split(' ')))
data['abbreviation_len'] = data['clean_abbreviation'] - data['text_len']

# stemming
data['text'] = data['text'].apply(stem)

# clean feature
data['text'] = data['text'].apply(clean_last)
data.drop(columns=['clean_emoji', 'clean_abbreviation'], inplace=True)

# 将title和text合并（主要目的是为了接下来更好地算TF值）
nd = data.copy()
for i in range(nd.shape[0]):
    text = nd.prompt_title.iloc[i] + ' ' + nd.text.iloc[i]
    nd['text'].iloc[i] = text
nd.drop(columns=['prompt_title'], inplace=True)

# 存储为csv文件
nd.to_csv('clean_data.csv', index=False)
