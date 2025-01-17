import re
import pandas as pd
import sys
sys.path.append('/opt/project/KG_Assist_LLM')

# 潘多拉数据清洗+kaggle 数据合并
data = pd.read_csv(
    '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-1.csv',index_col=0)

# %%
print(data.columns)

# %%
split_len = 500

# %%
data1 = data[data.body.map(len) <= split_len]
data2 = data[data.body.map(len) > split_len]

# %%
print(data1.shape[0], data2.shape[0])

# %%
data1.reset_index(drop=True, inplace=True)
data2.reset_index(drop=True, inplace=True)


# %%
data_temp = pd.DataFrame()
for i in range(data2.shape[0]):
    author = data2.loc[i, 'author']
    mbti = data2.loc[i, 'mbti']
    bodys = data2.loc[i, 'body'].split('. ')
    for body in bodys:
        if len(body.split(' ')) <= 5:
            continue
        temp = pd.DataFrame([[author, mbti, body]], columns=[
                            'author', 'mbti', 'body'])
        data_temp = pd.concat([data_temp, temp])


data_pand = pd.concat([data1, data_temp])


data = pd.read_csv('/opt/project/KG_Assist_LLM/data/MBTI/mbti.csv')


def data_process(txt: str):
    temp = []
    txt = txt.split('|||')
    for message in txt:
        website = re.findall('(https://\S+|http://\S+)', message)
        if not website:
            temp.append(message)
            continue
        for web in website:
            message = message.replace(web, '')
        temp.append(message) if message else None
    txt = "\n".join(temp)
    return txt


data.posts = data.posts.map(data_process)

data_temp = pd.DataFrame()
for i in range(data.shape[0]):
    if i == 500:
        break
    author = f'kaggle{i}'
    mbti = data.loc[i, 'type']
    bodys = data.loc[i, 'posts'].split('\n')
    for body in bodys:
        if len(body.split(' ')) <= 5:
            continue
        temp = pd.DataFrame([[author, mbti, body]], columns=[
                            'author', 'mbti', 'body'])
        data_temp = pd.concat([data_temp, temp])

data_pand = pd.concat([data_pand, data_temp])

data_pand['mbti'] = data_pand['mbti'].map(lambda x: str(x).upper())
data_pand.drop_duplicates(inplace=True)
data_pand.reset_index(drop=True, inplace=True)
print(data_pand.shape)
print(data_pand.head())
data_pand.to_csv(
    '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-2.csv', index=False)
