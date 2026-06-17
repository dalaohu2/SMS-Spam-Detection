import pandas as pd

# 这份数据有个老毛病：编码不是默认的，得指定 latin-1，否则会报错
df = pd.read_csv('spam.csv', encoding='latin-1')

# 它原始列名叫 v1 / v2，还带几列空的，我们只留有用的两列并重命名
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print(df.shape)              # 一共多少条
print(df.head())            # 看前几条长什么样
print(df['label'].value_counts())  # spam 和 ham 各多少
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 标签变数字：ham=0（正常），spam=1（诈骗）
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. X 是短信内容，y 是答案（0/1）
X = df['text']
y = df['label']

# 3. 分训练集 / 测试集，8:2
#    random_state=42：固定随机种子，保证你我的结果一致
#    stratify=y：让两边的 spam 比例都跟原始一样（还记得只有 13% 是 spam 吗）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF 上场
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)   # 训练文本：fit + transform
X_test_tfidf  = vectorizer.transform(X_test)        # 测试文本：只 transform

print(X_train_tfidf.shape)                          # (训练条数, 词汇表大小)
print(len(vectorizer.get_feature_names_out()))      # 一共学到多少个词

from sklearn.linear_model import LogisticRegression

# max_iter=1000：文本特征多，给它多点迭代次数，免得报"没收敛"的警告
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)          # 就这一行，开练

accuracy = model.score(X_test_tfidf, y_test)
print(accuracy)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test_tfidf)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# 真实的垃圾短信发送者常用的小把戏：字母换成形似的数字/符号
def disguise(text):
    text = text.lower()
    for a, b in [('o','0'), ('i','1'), ('e','3'), ('a','@'), ('s','$')]:
        text = text.replace(a, b)
    return text
# 效果：free→fr33, win→w1n, prize→pr1z3, call→c@ll, claim→cl@1m

# 1. 取出测试集里所有真·诈骗短信
spam_texts = X_test[y_test == 1]

# 2. 先量"原始版"模型能抓住几条（应该 ≈ 113，对应 recall 0.76）
original_caught = model.predict(vectorizer.transform(spam_texts)).sum()
print("原始 spam 抓住:", original_caught, "/", len(spam_texts))

# 3. 全部伪装一遍，再用【同一个】vectorizer transform（绝不能重新 fit！）
disguised = spam_texts.apply(disguise)
disguised_caught = model.predict(vectorizer.transform(disguised)).sum()
print("伪装后 spam 抓住:", disguised_caught, "/", len(disguised))