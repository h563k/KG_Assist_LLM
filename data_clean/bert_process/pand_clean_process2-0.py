# %%
import re
import pandas as pd

# %%
chunk_size = 10e6
chunks = pd.read_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process1.csv',chunksize=chunk_size)

# %%
def process(text:str):
        text = text.replace('...', '')
        website = re.findall(r'(\[.*?\]\(.*?\))', text)
        for web in website:
            text = text.replace(web, '')
        website = re.findall(r'(https://\S+|http://\S+)', text)
        for web in website:
            text = text.replace(web, '')
        website = re.findall(r'(<.*?>)', text)
        for web in website:
            text = text.replace(web, '')  
        text = text.strip('()')         
        text = text.split(' ')
        if len(text) <= 5:
            return None
        return ' '.join(text)

# %%
data_process = pd.DataFrame()
for i, chunk in enumerate(chunks):
    chunk = chunk[chunk['body'].notna()]
    chunk = chunk[['author', 'mbti', 'body']]
    chunk['body'] = chunk['body'].map(process)
    chunk = chunk[chunk['body'].notna()]
    data_process = pd.concat([data_process, chunk])
data_process.drop_duplicates(inplace=True)

# %%
data_process.to_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2.csv', index=False)

# %%
# chunk_test = chunk.groupby(by=['author','mbti']).agg(lambda x: list(x)) 
# chunk_test=chunk_test['body'].map(lambda x: '\n '.join(x))
# chunk_test

# %%
# chunk_test.to_csv('/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2.csv')


