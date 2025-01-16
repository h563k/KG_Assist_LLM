import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import torch

# 初始化BERT tokenizer和model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def data_process(task: str):
    temp = []
    task = task.split('|||')
    for message in task:
        website = re.findall('(https://\S+|http://\S+)', message)
        if not website:
            temp.append(message)
            continue
        for web in website:
            message = message.replace(web, '')
        temp.append(message) if message else None
    txt = "\n".join(temp)
    return txt


def get_embedding(text):
    text = data_process(text)
    # Tokenize 输入文本
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=128)
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用[CLS]标记的嵌入作为句子的嵌入
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return cls_embedding


def extract_dimension_labels(y, dimension):
    if dimension == 0:  # IE dimension
        return y.apply(lambda x: 0 if x[dimension] == 'I' else 1)
    elif dimension == 1:  # SN dimension
        return y.apply(lambda x: 0 if x[dimension] == 'S' else 1)
    elif dimension == 2:  # TF dimension
        return y.apply(lambda x: 0 if x[dimension] == 'T' else 1)
    elif dimension == 3:  # PJ dimension
        return y.apply(lambda x: 0 if x[dimension] == 'J' else 1)


# Load dataset
chunk_size = 1000
count = 0
for chunk in pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-3.csv', chunksize=chunk_size):
    if count == 50:
        break
    count += 1
data = chunk


# Preprocess data
X = data['body']
y = data['mbti']
X = X.apply(lambda x: get_embedding(x))
X = pd.DataFrame(X.tolist())


# Initialize variables to store F1 scores
f1_scores = []


def main():
    # Loop through each dimension
    for i, dimension in enumerate(['IE', 'SN', 'TF', 'PJ']):
        # Extract dimension labels
        y_dimension = extract_dimension_labels(data['mbti'], i)

        # Split data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_dimension, test_size=300)
        X_test = X[:300]
        y_test = y_dimension[:300]
        X_train = X[300:]
        y_train = y_dimension[300:]
        # Train XGBoost model
        model = xgb.XGBClassifier(objective='binary:logistic')
        # 创建SVM分类器
        model = SVC(kernel='linear')  # 也可以尝试 'rbf', 'poly' 等其他核函数
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate and store F1 score for the dimension
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
        print(f'F1 Score for {dimension}: {f1}')

    # Calculate and print average macro-F1 score
    average_macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f'Average Macro-F1 Score: {average_macro_f1}')


if __name__ == '__main__':
    main()