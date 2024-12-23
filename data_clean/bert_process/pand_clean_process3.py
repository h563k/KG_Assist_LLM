# %%
import pandas as pd


chunks = pd.read_csv(
    '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2.csv')

# %%
chunk = chunks.groupby(by=['author', 'mbti']).agg(lambda x: list(x))
chunk = chunk['body'].map(lambda x: '\n '.join(x))


# %%
chunk.to_csv(
    '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process3.csv')
