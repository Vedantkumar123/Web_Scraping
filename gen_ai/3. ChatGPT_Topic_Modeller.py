import pandas as pd
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import openai
import langchain
langchain.verbose = False
import os
from tqdm import tqdm
import time
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, AutoModel
import sys
import ScoreFunctions as sf

user = 'wr676jp'
sys.path.append(fr'C:\Users\{user}\OneDrive\Documents\SAGT')

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model_embedding = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)


def SentimentScore(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    score = sf.sentiment_score(scores)
    return score

folder_path = fr'C:\Users\{user}\EY\DATS Germany - TS-SA - TS-SA\03. Data\04. Final Output'
df = pd.read_excel(os.path.join(folder_path,'all_products_2023-08-15.xlsx'), sheet_name='Reviews')
df['Review ID'] = range(len(df))

df_run = df.copy()

# start = 0
# end = 10000
# df_run = df.loc[start:,:].copy()

# OpenAI Parameter - GPT-4 and GPT-4-32k turbo for testing
OPENAI_API_TYPE = 'azure'
OPENAI_API_BASE = 'https://frcdaoipocaoa02.openai.azure.com/' 
OPENAI_API_VERSION = '2023-03-15-preview'
OPENAI_API_KEY = '9c2f1a52118e4d1c99d3abd590fe34e4'

# Google Serp API
SERPER_API_KEY = "d4d674c45a86588928f879e9f2eecaff2e06c250"


gpt_model = "gpt-4"
temperature = 0


llm = AzureChatOpenAI(deployment_name=gpt_model,temperature=temperature,openai_api_key = OPENAI_API_KEY,openai_api_base = OPENAI_API_BASE,openai_api_version = OPENAI_API_VERSION,openai_api_type = OPENAI_API_TYPE)

answers = []
for x in tqdm(df_run['Review_Translated'], leave=True, position=0):
    try:
        country=x
        prompt="Which of the following topics (quality, design, price, feature) are being mentioned. Extract the sub-sentence for each topic. Answer by stating the topic: subsentence. {country}"
        prompt = PromptTemplate(input_variables=["country"],
                                template=prompt,
                            )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        answers.append(chain.run(countryntry=country))
    except:
        answers.append("Error")
        

df_run['Topic_Split'] = answers
df_run.to_excel(os.path.join(folder_path,"temp_gpt_output.xlsx"),index=False)

df_run['Topic_Split_2'] = df_run['Topic_Split'].str.split('\n')
df_test_2 = df_run[['Review ID','Topic_Split_2']].copy().explode('Topic_Split_2')
df_test_2['Topic_Split_3'] = np.where((df_test_2['Topic_Split_2'].str.lower().str.contains('not mention')) |\
                                      (df_test_2['Topic_Split_2'].str.lower().str.contains('none mention')) |\
                                      (df_test_2['Topic_Split_2'].str.lower().str.contains('no mention')) |\
                                      (df_test_2['Topic_Split_2'].str.lower().str.contains('n/a')) |\
                                      (df_test_2['Topic_Split_2'].str.lower().str.contains('without a specific sentence or context provided')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('Without a text or context provided')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The text ')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('None of the topics')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The question ')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The text does not provide')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The assistant cannot provide')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("You didn't provide a text or")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('Your question seems')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The assistant')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The sentence ')) |\
                                      (df_test_2['Topic_Split_2'] == ''),
                                        np.nan,df_test_2['Topic_Split_2'])
df_test_2['single_topic'] = df_test_2.groupby('Review ID')['Topic_Split_3'].transform('count')
df_test_3 = pd.concat([df_test_2[df_test_2['single_topic'] == 0],df_test_2[df_test_2['single_topic'] > 0].dropna()])
df_test_3['Topic'] = df_test_3['Topic_Split_3'].str.lower().str.extract('(quality|design|price|feature)')[0].str.title().fillna("Not Available")
output = df.copy().merge(df_test_3, on='Review ID')


output["Score"] = [SentimentScore(text) for text in tqdm(output['Topic_Split_3'].fillna(''),position=0,leave=True)]
output['Sentiment'] = np.where(output['Score'] < -0.33,"Negative",
                           np.where(output['Score'] < 0.33,"Neutral","Positive"))

output.to_excel(os.path.join(folder_path, "gpt_output.xlsx"))