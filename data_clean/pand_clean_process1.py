# %%
import pandas as pd
data = pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/author_profiles.csv')

# %%
data = data[['author', 'mbti']]

# %%
data_process = pd.DataFrame()

# %%
chunk_size = 10**4  # 每个chunk包含1百万行
chunks = pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/all_comments_since_2015.csv', chunksize=chunk_size)
for i, chunk in enumerate(chunks):
    chunk = chunk[['author', 'body']]
    data_merge = pd.merge(left=data, right=chunk, on='author', how='inner')
    print(data_merge.author.value_counts().shape)
    data_process = pd.concat([data_process, data_merge], axis=0)

# %%
data_process.to_csv('pand_clean_process1.csv')


