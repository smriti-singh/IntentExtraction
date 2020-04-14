#Python libraries used, libraries not installed can be installed using "pip install <libraryname>"
#Microsoft API website to try sample emails: https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/
import urllib
import json
from http.client import HTTPSConnection
import requests
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import statistics
import pyOutlook
from pyOutlook import OutlookAccount
import win32com
from win32com import client
import libpst



#Made a Text Analytics Service of Azure

#Our Base URL:
uri = "westcentralus.api.cognitive.microsoft.com"
#Our Access Key:
accessKey = ''

def GetSentiment (documents):
    #Path to the Sentiment Analysis Microsoft API:
    path = "/text/analytics/v2.0/sentiment" #sentiment HTTP/1.1
    headers = {'Ocp-Apim-Subscription-Key': accessKey}
    conn = HTTPSConnection (uri)
    body = json.dumps (documents)
    conn.request ("POST", path, body, headers)
    response = conn.getresponse ()
    return response.read ()

#Json format required by the API:
documents = { 'documents': [
    { 'id': '1', 'language': 'en', 'text': 'Hi team, So today I looked at your analysis the data seems to be working properly but I wish it was better. I would like a meeting of this sooner later this week. Let’s catch up and taco-about it' },
    { 'id': '2', 'language': 'en', 'text': 'I demand a report' }
]}
type(documents)



#Making the Email JSON---fix it
email_df=pd.read_csv(r"\positive and negative data.csv")

#Making auto matically dict:
keys=["id", "language", "text"]
dic={ 'documents':[{ key: 0 for key in keys}]}

#Making json:
x = {'documents': [{'id': 0, 'language': 0, 'text': 0}]}
json_data = json.dumps(x)
data = json.loads(json_data)

Example accessing Json:
data['documents'][1]
data['documents'][3].append({'id': '2', 'language': 'en', 'text': 'texuak'})


#Making email Json:
# emaildict={'documents': [{}]}
emaildict = {'documents': [{'id': '0', 'language': 'none', 'text': 'none'}]}
json_data = json.dumps(emaildict)
emaildata = json.loads(json_data)
# emaildata['documents'].append({'id': '2', 'language': 'en', 'text': 'texuak'})


# for index, row in email_df.iterrows():
for item_number in range(len(email_df)):
    emaildata['documents'].append({'id': str(email_df.loc[item_number]['id']) , 'language': str(email_df.loc[item_number]['language']) , 'text': str(email_df.loc[item_number]['text'])})
   
#Deleting the dummy entry:
if '0' in emaildata['documents'][0]['id']: 
    del emaildata['documents'][0]

#Result from API on Sample documents:
type(documents)
#Saving results generated in a Json:
result = GetSentiment(documents)
print("Resulting scores for each Document:", result)
#Saving json:
sentiments = json.loads(result)["documents"]
print(sentiments)

#Output on Email Data:
result = GetSentiment(emaildata)
print(result)
type(result)

#converting bytes result to dictionary:
answer=ast.literal_eval((result.decode('utf-8')))
type(answer)

#Accesing answer dictionary:
int(answer['documents'][8]['score'])



#Example adding IDs to each thread:
thread1=[]
for i in range(len(email_df)):
    if email_df['thread'][i]==3:
        thread1.append(email_df['id'][i])




#Makin a dictionary for all threads:
d={}
threads=[]
threads=email_df['thread'].unique()
more_keys=['x values-scores', 'y values-id', 'x avg-scores']
d={ threads:{ key:[] for key in more_keys} for threads in threads}

#Adding values inside this thread:
for key, value in d.items():
    for i in range(len(email_df)):
        if email_df['thread'][i]==key:
            d[key]['y values-id'].append(email_df['id'][i])
    for ids in d[key]['y values-id']:
        d[key]['x values-scores'].append(answer['documents'][ids-1]['score'])
      

#Dictionary ready for plotting each thread:
answer['documents']



#Adding Average Scores:
for key,value in d.items():
    for i, scores in enumerate(d[key]['x values-scores']):
        print(i)
        print(scores)
        if((i-1)<0):
            avg=d[key]['x values-scores'][i]
        else:
            avg=float((d[key]['x values-scores'][i] + avg )/2)
        print("Avg value", avg)
        d[key]['x avg-scores'].append(avg)
        print("appended avg score", avg, "for index", i)


for key, value in d.items():
    plt.plot(d[key]['y values-id'], d[key]['x avg-scores'], 'go--')
    plt.title('Thread'+ str(key))
    plt.ylabel('Scores for each Email')
    plt.xlabel('Individual emails in Thread' + str(key))
    plt.show()
