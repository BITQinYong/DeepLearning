#-×-coding:utf-8-*-
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
import pandas as pd
import tensorflow as tf
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
#import numpy as np
#from sklearn import metrics
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

subtrainLabel = pd.read_csv('csv/subtrainLabels.csv')
subtrainfeature = pd.read_csv("csv/3gramfeature.csv")
subtrain = pd.merge(subtrainLabel,subtrainfeature,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(subtrain,labels,test_size=0.4)

#RondomForest
srf = RF(n_estimators=500, n_jobs=-1)
srf.fit(X_train,y_train)
print "使用随机森林算法分类结果："
print srf.score(X_test,y_test)#随机森林
#svm
clf = SVC()
clf.fit(X_train,y_train)
print "使用支持向量分类算法分类结果："
print clf.score(X_test,y_test) #支持向量分类
#nusvm
clf=NuSVC()
clf.fit(X_train,y_train)
print "使用支持向量分类算法分类结果："
print clf.score(X_test,y_test)#核支持向量分类

clf = GaussianNB()
clf.fit(X_train,y_train)
print "使用朴素贝叶斯分类算法分类结果："
print clf.score(X_test,y_test)#朴素贝叶斯分类

classifier=LogisticRegression()
classifier.fit(X_train,y_train)
print "使用逻辑回归算法分类结果："
print classifier.score(X_test,y_test)#逻辑回归

classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
print "使用决策树算法分类结果："
print classifier.score(X_test,y_test)

classifier=GradientBoostingClassifier(n_estimators=200)
classifier.fit(X_train,y_train)
print "使用GBDT算法分类结果："
print classifier.score(X_test,y_test)

