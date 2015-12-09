#!/usr/local/bin/python

import os
import json
import pickle
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import *
import random
from collections import Counter

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def text_preprocessor(s):
	return s.lower().replace("-", " ").replace("_", " ")
			
def getData(vectorizer, useCached=True):
	print "vectorizing text..."
	train_file = os.path.join(".","train.json","train.json")
	with open(train_file) as data_file:
		train_data = json.loads(data_file.read())
	train_data_labels = [d["cuisine"] for d in train_data]
	test_file = os.path.join(".","test.json","test.json")
	with open(test_file) as data_file:
		test_data = json.loads(data_file.read())
	if useCached and (os.path.exists('train_vec.p') and os.path.exists('test_vec.p')):
		train_vec = pickle.load(open('train_vec.p', 'rb'))
		test_vec = pickle.load(open('test_vec.p', 'rb'))
	else:
		train_data_points = ['.\n'.join(d["ingredients"]) for d in train_data]
		test_data_points = ['.\n'.join(d["ingredients"]) for d in test_data]
		vec = vectorizer
		train_vec = vec.fit_transform(train_data_points)
		test_vec = vec.transform(test_data_points)
		pickle.dump(train_vec, open('train_vec.p', 'wb'))
		pickle.dump(test_vec, open('test_vec.p', 'wb'))
	
	return (train_vec, test_vec, train_data_labels, test_data)


def makePredictionMajority(model_list, data_point, printcount=20):
	results = []
	for model in model_list:
		#if printcount == 0:
		#	if model[1] != None: print "sampled features (first 5): " + str(model[1][:5])	
		if model[1] == None:
			pred = str(model[0].predict(data_point)[0])
		else:
			pred = str(model[0].predict(data_point[:,model[1]])[0])
		if pred != 'other': results.append(pred)
	if len(results) > 0:
		if printcount < 20:
			print str(results)
			print "chosen: " + Counter(results).most_common(1)[0][0]
		return Counter(results).most_common(1)[0][0]
	else:
		return 'null'


def trainBoostrapedModels(base_model, param_grid, param_list, dataset, data_labels, num_models, num_class_per_model, feature_sampling=None, resampleData=True):
	model_list = []
	distinct_labels = list(set(data_labels))
	for m in xrange(num_models+1):
		print "training model #" + str(m+1)
		sampled_feature = None
		
		if m == num_models:
			resampled_data = dataset
			refined_labels = data_labels
		else:
			if resampleData:
				resampled_data, resampled_label = resample(dataset, data_labels, replace=True)
			else:
				resampled_data = dataset
				resampled_label = data_labels
			
			if feature_sampling != None:
				sampled_feature = random.sample(range(0, resampled_data.shape[1]), int(resampled_data.shape[1]*feature_sampling))
				resampled_data = resampled_data[:,sampled_feature]
			
			if num_class_per_model == None:
				refined_labels = data_labels
			else:
				random.shuffle(distinct_labels)
				selected_labels = distinct_labels[:num_class_per_model]
				refined_labels = []
				for label in resampled_label:
					if label in selected_labels: refined_labels.append(label)
					else: refined_labels.append('other')
			
		if param_grid == None:
			if param_list != None:
				model_list.append((base_model().set_params(**param_list).fit(resampled_data, refined_labels), sampled_feature))
			else:
				model_list.append((base_model().fit(resampled_data, refined_labels), sampled_feature))
		else:
			model_list.append((GridSearchCV(base_model(), param_grid=param_grid, scoring='accuracy').fit(resampled_data, refined_labels), sampled_feature))
	return model_list

def calcAccuracy(labels, pred_labels):
	correct = 0
	for label, pred in zip(labels, pred_labels):
		if label == pred: correct += 1
	return float(correct)/float(len(labels))

submission_file = os.path.join(".","submission.csv")
unwanted_features = ['salt', 'water', 'onions', 'garlic']
train_vec, test_vec, train_data_labels, test_data = getData(TfidfVectorizer(analyzer='char', preprocessor=text_preprocessor, ngram_range=(2,3), max_df=0.5, stop_words=unwanted_features, strip_accents='unicode', tokenizer=LemmaTokenizer()), False)
dev_train_vec = train_vec[train_vec.shape[0]/3:,]
dev_train_label = train_data_labels[len(train_data_labels)/3:]
dev_test_vec = train_vec[:train_vec.shape[0]/3,]
dev_test_label = train_data_labels[:len(train_data_labels)/3]

#model_list = trainBoostrapedModels(base_model=LogisticRegression, param_grid=None, param_list={'C':11.0, 'penalty':'l2'}, dataset=train_vec, data_labels=train_data_labels, num_models=100, num_class_per_model=3)
model_list = trainBoostrapedModels(base_model=LogisticRegression, param_grid=None, param_list={'C':11.0, 'penalty':'l2'}, dataset=dev_train_vec, data_labels=dev_train_label, num_models=150, num_class_per_model=3, feature_sampling=0.6)

print "making predictions..."
predicted_results = []
#for i, data in enumerate(test_vec): predicted_results.append(makePredictionMajority(model_list, data, i))
for i, data in enumerate(dev_test_vec): predicted_results.append(makePredictionMajority(model_list, data, i))

print "calculating accuracy..."
print "accuracy: %.4f" % calcAccuracy(dev_test_label, predicted_results)

#print "printing submission..."
#writer = open(submission_file, 'w')
#writer.write('id,cuisine\n')
#for data, pred in zip(test_data, predicted_results):
#	writer.write(str(data["id"]) + "," + str(pred) + "\n")
#writer.close()

