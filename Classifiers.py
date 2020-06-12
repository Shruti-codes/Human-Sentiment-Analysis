import pandas as pd
from sklearn import metrics, cross_validation
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np


# Read feature list in a dataframe

df = pd.read_csv('features.csv')
data = df

# Logistic Regression Model

def LR_CV(data): #How to change your accuracy for matching: Change the C value below between 1e8 and  1e-8
   acc = []
   logreg = LogisticRegression(C=1e-6, multi_class='ovr', penalty='l2', random_state=0)
   predict = cross_validation.cross_val_predict(logreg, data.drop(['label'],axis=1), data['label'],cv=10)
   acc.append(accuracy_score(predict, data['label']))
   #print metrics.classification_report(data['label'], predict)
   F1 = metrics.f1_score(data['label'], predict)
   P = metrics.precision_score(data['label'], predict)
   R = metrics.recall_score(data['label'], predict)
   return (float(sum(acc) / len(acc))) * 100, F1 * 100, P * 100, R * 100

# SVM Model

def SVM_CV(data):
    acc = []
    SVM = SVC(C=0.1, kernel='linear')
    predict = cross_validation.cross_val_predict(SVM, data.drop(['label'],axis=1), data['label'],cv=10)
    acc.append(accuracy_score(predict, data['label']))
    #print metrics.classification_report(data['label'], predict)
    F1 = metrics.f1_score(data['label'], predict)
    P = metrics.precision_score(data['label'], predict)
    R = metrics.recall_score(data['label'], predict)
    return (float(sum(acc) / len(acc))) * 100, F1 * 100, P * 100, R * 100

# Decision Tree model   
   
def DT_CV(data):
    acc = []
    classifier = DecisionTreeClassifier()
    predict = cross_validation.cross_val_predict(classifier, data.drop(['label'], axis=1), data['label'], cv=10)
    acc.append(accuracy_score(predict, data['label']))
    #print metrics.classification_report(data['label'], predict)
    F1 = metrics.f1_score(data['label'], predict)
    P = metrics.precision_score(data['label'], predict)
    R = metrics.recall_score(data['label'], predict)
    return (float(sum(acc) / len(acc))) * 100, F1 * 100, P * 100, R * 100

# Naive Bayes Model
   
def NB_CV(data):
    acc = []
    classifier = GaussianNB()
    predict = cross_validation.cross_val_predict(classifier, data.drop(['label'], axis=1), data['label'], cv=10)
    acc.append(accuracy_score(predict, data['label']))
    #print metrics.classification_report(data['label'], predict)
    F1 = metrics.f1_score(data['label'], predict)
    P = metrics.precision_score(data['label'], predict)
    R = metrics.recall_score(data['label'], predict)
    return (float(sum(acc) / len(acc)))*100, F1*100, P*100, R*100

# Neural Network Model
   
def NN_CV(data):
    acc = []
    classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    predict = cross_validation.cross_val_predict(classifier, data.drop(['label'], axis=1), data['label'], cv=10)
    acc.append(accuracy_score(predict, data['label']))
    #print metrics.classification_report(data['label'], predict)
    F1 = metrics.f1_score(data['label'], predict)
    P = metrics.precision_score(data['label'], predict)
    R = metrics.recall_score(data['label'], predict)
    return (float(sum(acc) / len(acc))) * 100, F1 * 100, P * 100, R * 100

# Random Forest Model
   
def RandForest_CV(data):
    acc = []
    classifier = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    predict = cross_validation.cross_val_predict(classifier, data.drop(['label'], axis=1), data['label'], cv=10)
    acc.append(accuracy_score(predict.round(), data['label']))
    #print metrics.classification_report(data['label'], predict.round())
    F1 = metrics.f1_score(data['label'], predict.round())
    P = metrics.precision_score(data['label'], predict.round())
    R = metrics.recall_score(data['label'], predict.round())
    return (float(sum(acc) / len(acc))) * 100, F1 * 100, P * 100, R * 100

features = ['User mention', 'Exclamation', 'Question mark', 'Ellipsis', 'Interjection', 'UpperCase', 'RepeatLetters',
       'SentimentScore', 'positive word count', 'negative word count', 'polarity flip', 'Nouns', 'Verbs',
            'PositiveIntensifier', 'NegativeIntensifier', 'Bigrams', 'Trigram', 'Skipgrams', 'Emoji Sentiment',
            'Passive aggressive count','Emoji_tweet_polarity flip']

# Calculate the accuracies for the required model

print "Model: " + "LR"
for feature in features:
    tiny_data = data[[feature, 'label']]
    Acc, F1, P, R = LR_CV(tiny_data)
    print feature
    print "Acc: "+str(Acc)+" F1: "+str(F1)+ " P: "+str(P)+" R: "+str(R)