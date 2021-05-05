'''
Declaration :
I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of F for the course for any additional offense.
-Priyanka Prakash Tanpure
'''

import os
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import math
import random

nltk.download('stopwords')

path_spam = "train/spam/"
path_ham = "train/ham/"

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

totalspamwordcount = 0
totalspamfilecount = 0
totalhamwordcount = 0
totalhamfilecount = 0
spam_words_list =[]    #spam word list for each file
ham_words_list=[]     #ham word list for each file
totalCorrectPredSpam = 0;
totalCorrectPredHam = 0;
weightVector ={}

default_weight = 0.1

threshold_list = [3,9,15,20,25,30]
learning_rate = [0.08,0.1,0.12,0.14,0.16,0.2]


def processfiles(file_path,is_stop_param):
    f= open(file_path,'r',errors='ignore')
    #text = f.read().lower()
    wordlist = list()
    for line in f:
        line = line.rstrip()
        words = re.sub('[^a-zA-Z0-9 ]','',line).split()
        for word in words:
            wordlist.append(word)
    wordlist = [stemmer.stem(word) for word in list(filter(None,wordlist))]
    if is_stop_param == True:
        wordlist = [word for word in wordlist if not word in stop_words]
    return wordlist

def resetWeightVector():
    for word in weightVector:
        weightVector[word] = default_weight
        weightVector["BIAS"]  = default_weight

def buildPerceptronModel(is_stop_param):
    
    global spam_words_list,ham_words_list,weightVector
    
    #create a weightVector directory for all the words from ham and spam files word:weight initialize with default weight 
    for filepath in [path_spam,path_ham]:
        for file in os.listdir(filepath):
            if file.endswith(".txt"):
                file_path = f"{filepath}{file}"
                wordlist = processfiles(file_path,is_stop_param)
                for word in wordlist:
                    if word not in weightVector:
                        weightVector[word] = default_weight    #initialize with random value
    
    weightVector["BIAS"]  = default_weight     #Bias term w0*x0 (x0=1)
            
    #spam words list of each file
    for file in os.listdir(path_spam):
        if file.endswith(".txt"):
            file_path = f"{path_spam}{file}"
            wordlistspam = processfiles(file_path,is_stop_param)
            spam_words_list.append(wordlistspam)


    #ham words list of each file
    for file in os.listdir(path_ham):
        if file.endswith(".txt"):
            file_path = f"{path_ham}{file}"
            wordlistham = processfiles(file_path,is_stop_param)
            ham_words_list.append(wordlistham)

    

def predict(feature):
    global weightVector
    summation = sum(feature[word] * weightVector[word] for word in feature)
    summation +=weightVector["BIAS"]
    if summation > 0:
        return 1
    else:
        return -1
    
def wordCount(wordlist):
    feature = {}
    for word in wordlist:
        if word in feature:
            feature[word] +=1
        else:
            feature[word] =1
            
    return feature

    
def trainPerceptronModel(threshsold,learning_rate):
    global spam_words_list,ham_words_list,weightVector
    
    for _ in range(threshsold):
        for spamwordlist in spam_words_list:
            feature_spam = wordCount(spamwordlist)
            prediction = predict(feature_spam)
            for word in feature_spam:
                weightVector[word] += learning_rate*(1-prediction)*feature_spam[word]
                weightVector["BIAS"] += learning_rate*(1-prediction)
            
        for hamwordlist in ham_words_list:
            feature_ham = wordCount(hamwordlist)
            prediction = predict(feature_ham)
            for word in feature_ham:
                weightVector[word] += learning_rate*(-1-prediction)*feature_ham[word]
                weightVector["BIAS"] += learning_rate*(-1-prediction)
    
    
def modelAccuracy(is_stop_param):
    global weightVector
    path_spam_test = "test/spam/"
    path_ham_test = "test/ham/"
    correctpred=0
    total = 0
    accuracy =0
    for filepath in [path_spam_test,path_ham_test]:
        for file in os.listdir(filepath):
            if file.endswith(".txt"):
                file_path = f"{filepath}{file}"
                wordlist = processfiles(file_path,is_stop_param)
                for word in wordlist:
                    if word not in weightVector:
                        weightVector[word] = default_weight
    
    for file in os.listdir(path_spam_test):
        if file.endswith(".txt"):
            file_path = f"{path_spam_test}{file}"
            wordlistspam = processfiles(file_path,is_stop_param)
            feature_spam = {}
            for word in wordlistspam:
                if word in feature_spam:
                    feature_spam[word] +=1
                else:
                    feature_spam[word] =1
            
            prediction = predict(feature_spam)            
            if prediction > 0:
                correctpred +=1
            total+=1
    
    for file in os.listdir(path_ham_test):
        if file.endswith(".txt"):
            file_path = f"{path_ham_test}{file}"
            wordlistham = processfiles(file_path,is_stop_param)
            feature_ham = {}
            for word in wordlistham:
                if word in feature_ham:
                    feature_ham[word] +=1
                else:
                    feature_ham[word] =1
            
            prediction = predict(feature_ham)
            if prediction <= 0:
                correctpred +=1
            total +=1
            
    accuracy = correctpred/total
    return accuracy
            
buildPerceptronModel(True)

print("Without Stop Words")
for lr in learning_rate:
    for threshold in threshold_list:
        resetWeightVector()
        trainPerceptronModel(threshold,lr)
        accuracy = modelAccuracy(True)
        print("Iterations : {}\t learning_rate : {}\t accuracy: {} ".format(threshold,lr,accuracy))
        
buildPerceptronModel(False)
print("\n\nWith Stop Words")
for lr in learning_rate:
    for threshold in threshold_list:
        resetWeightVector()
        trainPerceptronModel(threshold,lr)
        accuracy = modelAccuracy(False)
        print("Iterations : {}\t learning_rate : {}\t accuracy: {} ".format(threshold,lr,accuracy))