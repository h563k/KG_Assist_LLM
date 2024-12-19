# %%
import sys
import time
sys.path.append('/opt/project/KG_Assist_LLM')

# %%
import pandas as pd
from functionals.llm_api import openai_response

# %%
chunk_size = 10
chunks = pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-2-2.csv', chunksize=chunk_size)

# %%
system_prompt = "Determine if the following sentence involves character traits. If it does not, just respond with 'No'. If it does, just respond with 'Yes'"

# %%
data_process = pd.DataFrame()
left, right = 21, 2000
for j, chunk in enumerate(chunks):
    chunk.dropna(inplace=True)
    if left <= j <= right:
        print(j)
        temp = chunk.copy()
        temp.reset_index(inplace=True, drop=True)
        temp.loc[:, 'is_mbti'] = None
        for i in range(int(chunk.shape[0])):
            author = temp.iloc[i]['author']
            mbti = temp.iloc[i]['mbti']
            body = temp.iloc[i]['body']
            if not body:
                continue
            if len(body.split(' ')) < 5:
                continue
            response = openai_response(system_prompt=system_prompt,
                                       prompt=body,
                                       openai_type='openai_origin',
                                       )
            temp.loc[i, 'is_mbti'] = response
        data_process = pd.concat([data_process, temp], axis=0)
        data_final = data_process.copy()
        data_final.reset_index(drop=True, inplace=True)
        data_final.to_csv(
            '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-3.csv')
    elif j <= left:
        continue
    else:
        break
