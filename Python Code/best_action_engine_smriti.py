#Import Libraies:
from http.client import HTTPSConnection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib
import json
import requests
import ast
import statistics
import csv
import ast


#Functions:
def preproc_csv(csv_data):
    emaildict = {'documents': [{'id': '0', 'language': 'none', 'text': 'none'}]}
    json_data = json.dumps(emaildict)
    emaildata = json.loads(json_data)
    for item_number in range(len(csv_data)):
        emaildata['documents'].append({'id': str(csv_data.loc[item_number]['id']) , 'language': str(csv_data.loc[item_number]['language']) , 'text': str(csv_data.loc[item_number]['text'])})
    #Deleting the dummy entry:
    if '0' in emaildata['documents'][0]['id']: 
        del emaildata['documents'][0]
    return emaildata

def preproc_json(json_file):
    #Read Json file-Deserialize Json:
    with open(json_file, "r") as read_file:
        json_loaded_data = json.load(read_file)
    emaildict = {'documents': [{'id': '0', 'language': 'none', 'text': 'none'}]}
    json_data = json.dumps(emaildict)
    emaildata = json.loads(json_data)
    for item_number in range(len(json_loaded_data)):
        emaildata['documents'].append({'id': str(json_loaded_data[item_number]['id']) , 'language': str(json_loaded_data[item_number]['language']) , 'text': str(json_loaded_data[item_number]['text'])})
    #Deleting the dummy entry:
    if '0' in emaildata['documents'][0]['id']: 
        del emaildata['documents'][0]
    return emaildata


def getSentiment(document, subscription_key, endpoint):
    sentiment_api_url = endpoint + "/text/analytics/v2.1/sentiment"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.post(sentiment_api_url, headers=headers, json=document)
    result_json = response.json()
    return result_json

def getEntities(document, subscription_key, endpoint):
    sentiment_api_url = endpoint + "/text/analytics/v3.0-preview.1/keyPhrases"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.post(sentiment_api_url, headers=headers, json=document)
    result_json = response.json()
    return result_json


#Main:
#input Data:
df=pd.read_csv(r"\redmi6.csv", encoding="ISO-8859-1")
df["text"]= df["Review Title"]+ " " + df["Comments"]
df=df[["id", "language", "Category", "text"]]
df.to_csv(r"\sentiment_analysis_preprocessed_data_feedback.csv")
df.Category.unique()

#For each Type of category:
display_df=df[df.Category=="Display"]
others_df=df[df.Category=="Others"]
camera_df=df[df.Category=="Camera"]
battery_df=df[df.Category=="Battery"]
delivery_df=df[df.Category=="Delivery"]

display_df.to_csv(r"\sentiment_analysis_preprocessed_data_feedback_display.csv")
#Example for display:

#Making json:
#for display:
csv_file=r"\sentiment_analysis_preprocessed_data_feedback_display.csv"
json_file_example=r"\text_feedback_data_json_display.json"
data_document=pd.read_csv(csv_file)


#Reading CSV and Saving to json
f = open(csv_file, 'rU' )  
columns_names=list(data_document.columns.values)
# Change each fieldname to the appropriate field name. I know, so difficult.  
reader = csv.DictReader( f, fieldnames = columns_names)  
next(reader, None)
# Parse the CSV into JSON  
out = json.dumps( [ row for row in reader ] )  
print("JSON parsed!")  
# Save the JSON  
f = open(json_file_example, 'w')  
f.write(out)  
print("JSON saved!") 

#preprcessed:
pre_processed_json_data=preproc_json(json_file_example)


pre_processed_data=preproc_csv(data_document)
type(pre_processed_data)

#Sentiment analysis:
subscription_key = ""
endpoint = "https://textanalyticssmriti.cognitiveservices.azure.com/"

sentiments=getSentiment(pre_processed_data, subscription_key, endpoint)


positive_display=[]
negative_display=[]
df
for item in sentiments['documents']:
    if item['score']>0.6:
        positive_display.append(int(item['id']))
    else:
        negative_display.append(int(item['id']))


postitive_df=df[df['id'].isin(positive_display)]
negative_df=df[df['id'].isin(negative_display)]

negative_df.to_csv(r"\negative_display_feedback.csv")

#Pre-processing for negative display feedback:
#for display:
csv_file=r"\negative_display_feedback.csv"
json_file_example=r"\negative_display_feedback_json.csv"
data_document=pd.read_csv(csv_file)


#Reading CSV and Saving to json
f = open(csv_file, 'rU' )  
columns_names=list(data_document.columns.values)
# Change each fieldname to the appropriate field name. I know, so difficult.  
reader = csv.DictReader( f, fieldnames = columns_names)  
next(reader, None)
# Parse the CSV into JSON  
out = json.dumps( [ row for row in reader ] )  
print("JSON parsed!")  
# Save the JSON  
f = open(json_file_example, 'w')  
f.write(out)  
print("JSON saved!") 

#preprcessed:
pre_processed_json_data=preproc_json(json_file_example)


pre_processed_data=preproc_csv(data_document)

#Get entities from pre-processed negative display data:

entities=getEntities(pre_processed_data, subscription_key, endpoint)

#Adding all entity output in df
df_intent=pd.DataFrame(columns=['id', 'extracted_intents'])
for item in entities['documents']:
    df_intent=df_intent.append({'id': item['id'], 'extracted_intents': str(item['keyPhrases']) }, ignore_index=True)

df_intent.to_csv(r"\list.csv")
#Converting intent into document:
document=[]
for item in entities['documents']:
    document.append(str(item['keyPhrases']))


#Clustering text:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

#Word2vec embedding:

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
document

def Convert(string): 
    li = list(string.split(" ")) 
    return li 


new_doc=[]
for sentence in document:
    sentence=Convert(sentence)
    new_doc.append(sentence)

newlist = [word for line in mylist for word in line.split()]

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],['this', 'is', 'the', 'second', 'sentence'],['yet', 'another', 'sentence']]

# train model
model = Word2Vec(new_doc, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


#Another try:
import gensim
from sklearn.manifold import TSNE
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()




#New try word2vec, tsne, google:
import re
import codecs
import nltk
import multiprocessing
from gensim.models import Word2Vec


def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def prepare_for_w2v(filename_from, filename_to, lang):
    raw_text = codecs.open(filename_from, "r", encoding='windows-1251').read()
    with open(filename_to, 'w', encoding='utf-8') as f:
        for sentence in nltk.sent_tokenize(raw_text, lang):
            print(preprocess_text(sentence.lower()), file=f)


def train_word2vec(filename):
    data = gensim.models.word2vec.LineSentence(filename)
    return Word2Vec(data, size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())

keys = ['Camera', 'Battery', 'Metal', 'Screen', 'Notification']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


import matplotlib.pyplot as plt
import matplotlib.cm as cm


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Most Occuring Customer Concerns for Device', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

#Doesn't work for sentences:
#Use Doc2vec:
import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec

train= pd.read_csv(r"list_v2.csv")
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content_train = []
j=0
for em in train['extracted_intents'].values:
    all_content_train.append(LabeledSentence1(em,[j]))
    j+=1
print("Number of texts processed: ", j)

d2v_model = Doc2Vec(all_content_train)
d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)

kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100) 
X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)

#old viz:
pyplot.scatter(datapoint[:, 0], datapoint[:, 1])
sent = list(d2v_model.docvecs.doctag_syn0)
d2v_model.docvecs.doctag_syn0

for i, sent in enumerate(sent):
    pyplot.annotate(sent, xy=(datapoint[i, 0], datapoint[i, 1]))
pyplot.show()




import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

