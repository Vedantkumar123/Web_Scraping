from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, AutoModel
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import sys
import os
import openai
import langchain
langchain.verbose = False



def sentiment_score(scores):
    """
    This function takes a list of sentiment scores, calculates the softmax of
    the scores, and identifies the sentiment with the highest probability.
    It then clips the score to ensure it falls within predefined bounds
    for positive, negative, and neutral sentiments.
    
    Original Bound -> OB
    Desired Bound -> DB

    new score = DB[lower] + (DB[upper] - DB[lower]) * (current score - OB[lower])
                             ---------------------
                            (OB[upper] - OB[lower])

    note: 
    There is some discontinuity in the calculation of the scores.
    for example:
        for a neutral label with sentiment score of -0.33 it mean there is no positive influence.
        but for negative label with negative score of -0.92 there might be some positive influence.

    """
    def clip_score(original_score, desired_bound, original_bound):
        """."""
        normalized_difference = (original_score - original_bound['lower'])\
            / (original_bound['upper'] - original_bound['lower'])
        score = desired_bound['lower'] + normalized_difference *\
            (desired_bound['upper'] - desired_bound['lower'])
        return score

    softmax_scores = softmax(scores)
    max_ind = softmax_scores.argmax()

    neutral_desired_bound = {'lower': -0.20, 'upper': 0.20}
    positive_desired_bound = {'lower': neutral_desired_bound['upper']+0.01, 'upper': 1}
    negative_desired_bound = {'lower': -1, 'upper': neutral_desired_bound['lower']-0.01}

    if max_ind == 2:
        pos_score = softmax_scores.max();
        score = clip_score(pos_score, positive_desired_bound, {'lower': 0.33, 'upper': 1})
    elif max_ind == 0:
        neg_score = -1*softmax_scores.max()
        score = clip_score(neg_score, negative_desired_bound, {'lower': -1, 'upper': -0.33})
    else:
        neutral_score = softmax_scores[2]-softmax_scores[0]
        score = clip_score(neutral_score, neutral_desired_bound, {'lower': -0.49, 'upper': 0.49})

    return score


def sentimentScore(text):
    '''
    text: text to calculate the sentiment for
    '''
    if(str(text) == "nan"):
        return None
    
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, stride=5, return_overflowing_tokens=True, padding = True)
    
    num_splits = encoded_input["input_ids"].shape[0]
    sliding_scores = np.zeros((num_splits,3))
    
    for i in range(0, num_splits):
        output = model(input_ids = encoded_input["input_ids"][i].unsqueeze(0), attention_mask = encoded_input["attention_mask"][i].unsqueeze(0))
        score = output[0][0].detach().numpy()
        sliding_scores[i] = score

    scores = np.average(sliding_scores, axis=0)
    score = sentiment_score(scores)
    return score


#### Model Definition ####
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model_name_embedding = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model_embedding = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model_embedding = SentenceTransformer(model_name_embedding)


## ChatGPT Set up
# OpenAI Parameter - GPT-4 and GPT-4-32k turbo for testing
OPENAI_API_TYPE='azure'
OPENAI_API_BASE='https://frcdaoipocaoa02.openai.azure.com/' 
OPENAI_API_VERSION='2023-03-15-preview'
OPENAI_API_KEY='9c2f1a52118e4d1c99d3abd590fe34e4'

# Google Serp API
SERPER_API_KEY = "d4d674c45a86588928f879e9f2eecaff2e06c250"


gpt_model="gpt-4"
temperature=0

# Initialize LLM 
llm = AzureChatOpenAI(deployment_name=gpt_model,temperature=temperature,openai_api_key = OPENAI_API_KEY,openai_api_base = OPENAI_API_BASE,openai_api_version = OPENAI_API_VERSION,openai_api_type = OPENAI_API_TYPE)


#### Up to here, it was all data model preparation 


folder_path = r'C:\Users\WN369BP\EY\TD - Advanced Analytics - General\03_Output_Files' # Replace with folder
df = pd.read_excel(os.path.join(folder_path,'output_translated_cleaned.xlsx'))


answers = []
for x in tqdm(df['Text'],leave=True,position=0):
    if x == "Text not found":
        answers.append("No text passed")
    elif len(x) > 550:
        answers.append("Text is too long")
    else:
        try:
            country=x
            prompt="Based on customer reviews, which of the following topics (customer service, store location, products, store tmosphere) are being mentioned. Extract the sub-sentence only for the topics mentioned. Answer by stating the topic: subsentence. {country}"
            prompt = PromptTemplate(input_variables=["country"],
                                    template=prompt,
                                )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            
            answers.append(chain.run(country=country))
        except:
            answers.append("Error")
        

df['Topic_Split'] = answers
df.to_excel(os.path.join(folder_path,"temp_gpt_output.xlsx"),index=False)

df['Topic_Split_2'] = df['Topic_Split'].str.split('\n')
df = df.reset_index()
df_test_2 = df[['index','Topic_Split_2']].copy().explode('Topic_Split_2')
df_test_2['Topic_Split_3'] = np.where((df_test_2['Topic_Split_2'].str.lower().str.contains('not mention')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('Without a text or context provided')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The text ')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('None of the topics')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The question ')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The text does not provide')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The assistant cannot provide')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("You didn't provide a text or")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('Your provided')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The assistant')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith('The sentence ')) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("I'm sorry, but as an AI")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("To provide an accurate response,")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("Based on the")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("No text passed")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("I'm sorry")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("Text is too long")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("The provided text")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("If you ")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("Please provide")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("Your request")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("For a more")) |\
                                      (df_test_2['Topic_Split_2'].str.startswith("If this text")) |\
                                      (df_test_2['Topic_Split_2'] == ''),
                                        np.nan,df_test_2['Topic_Split_2'])
df_test_2['single_topic'] = df_test_2.groupby('index')['Topic_Split_3'].transform('count')
df_test_3 = pd.concat([df_test_2[df_test_2['single_topic'] == 0],df_test_2[df_test_2['single_topic'] > 0].dropna()])



## Need to replace here with the new topics (they must match the ones from the prompt (line 132)) ##

df_test_3['Topic'] = df_test_3['Topic_Split_3'].str.lower().str.extract('(products|customer service|store atmosphere|store location|price)')[0].str.title().fillna("Not Available")
output = df.drop(columns=['Topic_Split','Topic_Split_2']).copy().merge(df_test_3.drop(columns=['Topic_Split_2','single_topic']),
                                   on='index')
output["Score"] = [sentimentScore(text) for text in tqdm(output['Topic_Split_3'].fillna(''),position=0,leave=True)]
output['Sentiment'] = np.where(output['Score'] < -0.33,"Negative",
                            np.where(output['Score'] < 0.33,"Neutral","Positive"))

output.to_excel(os.path.join(folder_path,"gpt_output_without_cleaning.xlsx"))