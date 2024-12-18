import sys
sys.path.append('/opt/project/KG_Assist_LLM')
import pandas as pd

# %%
data = pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-1.csv')

# %%
print(data.columns)

# %%
split_len = 500

# %%
data1 = data[data.body.map(len)<=split_len]
data2 = data[data.body.map(len)>split_len]

# %%
print(data1.shape[0], data2.shape[0])

# %%
data1.reset_index(drop=True, inplace=True)
data2.reset_index(drop=True, inplace=True)

# %%
data1.to_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-2-1.csv', index=False)

# %%
data_temp = pd.DataFrame()
for i in range(data2.shape[0]):
    author = data2.loc[i, 'author']
    mbti = data2.loc[i, 'mbti']
    bodys = data2.loc[i, 'body'].split('. ')
    for body in bodys:
        temp = pd.DataFrame([[author, mbti, body]], columns=['author', 'mbti', 'body'])
        data_temp = pd.concat([data_temp, temp])

# %%
data_temp.to_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-2-2.csv', index=False)


