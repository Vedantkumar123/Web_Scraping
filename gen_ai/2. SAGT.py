import sys
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from datetime import datetime
import ScoreFunctions as sf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

pd.options.mode.chained_assignment = None
logging.set_verbosity_error()

#sys.path.append(r'C:\Users\WN369BP\OneDrive - EY\Documents\SAGT') #Change to your local git folder

def sentimentScore(text):
    '''
    text: text to calculate the sentiment for
    '''
    if(str(text) == "nan"):
        return None
    
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, stride=5, padding='max_length', return_overflowing_tokens=True)
    
    num_splits = encoded_input["input_ids"].shape[0]
    sliding_scores = np.zeros((num_splits,3))
    
    for i in range(0, num_splits):
        output = model(input_ids = encoded_input["input_ids"][i].unsqueeze(0), attention_mask = encoded_input["attention_mask"][i].unsqueeze(0))
        score = output[0][0].detach().numpy()
        sliding_scores[i] = score

    scores = np.average(sliding_scores, axis=0)
    score = sf.sentiment_score(scores)
    return score


def getTopics(text,topics_emb,thresholds): #Do the embedding of the topics only once and outisde the function
    '''
    text: text to assign a topic to
    topics_emb: a series of keywords embedding for topics
    thresholds: a series from POF Topics.xlsx containing thresholds
    '''
    if(str(text) == "nan"):
        return [None, 'other']
    
    embedded_review = model_embedding.encode(text)
    output = {}
    for x in topics_emb.index: 
        output[x] = cosine_similarity(topics_emb[x].reshape(1,-1), embedded_review.reshape(1,-1))[0][0]

    topic = sorted(output.items(), key=lambda i:i[1], reverse=True)[0][0]
    similarity = output[topic]
    
    if similarity > thresholds.get(topic):
        return [similarity, topic]
    else:
        return [None, 'other']
    

#### Model Definition ####
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# model_name_embedding = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# model_embedding = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# model_embedding = SentenceTransformer(model_name_embedding)

def runSAGT(user, part_size = 1000, part_start = 1, part_end = 2):
    '''
    user: seven character user name of format CCDDDCC for file paths
    part_size: number of rows for each part
    part_start: number of the first part, included
    part_end: number of the last part, excluded
    '''
    #### Read Base Data ####
    data_path = fr"C:\Users\{user}\EY\DATS Germany - TS-SA - TS-SA\03. Data"
    reviews_file = data_path + r"\04. Final Output\all_products_2023-08-15.xlsx"
    topics_file = data_path + r"\05. Topic Definition\POF Topics.xlsx"
    base_df = pd.read_excel(reviews_file, sheet_name='Reviews', usecols = ['UID','Review_Cleaned']) #Product ID & Review Text

    for i in tqdm(range(part_start,part_end)):
        #### Input and Output files ####
        part = i
        output_file = data_path + r"\04. Final Output\Results " + datetime.today().strftime('%Y-%m-%d') + " Part " + str(part) + ".xlsx"

        #### Framework ####
        df_part = base_df.loc[((part-1)*part_size):(part*part_size)-1]
        # df_topics = pd.read_excel(topics_file, sheet_name='Topics', index_col='Topics')
        # topics_emb = df_topics['Keywords'].map(model_embedding.encode)

        df_part["Score"] = [sentimentScore(text) for text in tqdm(df_part['Review_Cleaned'])]
        df_part['Sentiment'] = np.where(df_part['Score'] < -0.33,"Negative",
                                np.where(df_part['Score'] > 0.33,"Positive","Neutral"))
        # df_part[['Similarity','Topic']] = [getTopics(text,topics_emb,df_topics['Thresholds']) for text in tqdm(df_part['Review_Cleaned'], total=len(df_part['Review_Cleaned']))] 

        #### Output ####
        df_part.to_excel(output_file, sheet_name="Results")


runSAGT("HJ283SL")

#Models
#
#https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    