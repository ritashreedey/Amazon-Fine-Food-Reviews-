# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 23:56:35 2021

@author: Vipul Gaurav
"""
from flask import Flask, render_template, request
import pickle
from bs4 import BeautifulSoup
import re

def decontracted(phrase):
	
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
			"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
			'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
			'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
			'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
			'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
			'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
			'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
			'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
			'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
			's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
			've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
			"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
			"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
			'won', "won't", 'wouldn', "wouldn't"])

def clean_text(sentance):
    # Remove HTML Tags
	sentance = re.sub(r"http\S+", "", sentance)
    # Decontract the text
	sentance = BeautifulSoup(sentance, 'lxml').get_text()
	sentance = decontracted(sentance)
    # Remove extra spaces and digits
	sentance = re.sub("\S*\d\S*", "", sentance).strip()
    # Remove non-alphanumeric
	sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # Lowercase the sentences
	sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
	return sentance.strip()



# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'Naive-Bayes.pkl'
model = pickle.load(open(filename, 'rb'))
vect= pickle.load(open('Count-Vectorizer.pkl','rb'))

app = Flask(__name__, template_folder='template') 


@app.route('/',methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		review_text= decontracted(message)
		review_text= clean_text(message)
		test_vect  = vect.transform(([review_text]))
		my_prediction = model.predict(test_vect)
		return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

