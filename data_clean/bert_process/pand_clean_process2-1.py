# %%
from functionals.llm_api import openai_response
import pandas as pd
import sys
sys.path.append('/opt/project/KG_Assist_LLM')

# %%

# %%
system_prompt = "Determine if the following sentence involves character traits. If it does not, just respond with 'No'. If it does, just respond with 'Yes'"

# %%
chunk_size = 100
chunks = pd.read_csv(
    '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2.csv', chunksize=chunk_size)

# %%
data_process = pd.DataFrame()
for j, chunk in enumerate(chunks):
    print(j)
    if 109 <= j <= 201:
        temp = chunk.copy()
        temp.reset_index(inplace=True, drop=True)
        temp.loc[:, 'is_mbti'] = None
        for i in range(int(chunk_size)):
            author = temp.iloc[i]['author']
            mbti = temp.iloc[i]['mbti']
            body = temp.iloc[i]['body']
            response = openai_response(system_prompt=system_prompt,
                                       prompt=body,
                                       openai_type='openai_origin',
                                       )
            temp.loc[i, 'is_mbti'] = response
        data_process = pd.concat([data_process, temp], axis=0)
        data_final = data_process.copy()
        data_final.reset_index(drop=True, inplace=True)
        data_final.to_csv(
            '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-1.csv')
    elif j <= 108:
        continue
    else:
        break
