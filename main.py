#!/usr/bin/env python
# coding: utf-8

# In[44]:
from os.path import dirname, join

import matplotlib.pyplot as plt
import librosa 
import os
import numpy as np
import pandas as pd
import timeit
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  MinMaxScaler
import heapq
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
'''
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
'''
import warnings
warnings.filterwarnings("ignore")




scaler=MinMaxScaler()
tr_f=os.listdir('/home/datta/Downloads/UrbanSound/urban-sound-classification/train/Train/')
#tr_f=os.listdir('train/Train1/')

#print(tr_f[0])

print('Number of clips used for training are:{}'.format(int(len(tr_f)*0.85)))
#tr_f.sort()
fstart=timeit.default_timer()

#print('\nShowing First 5 of training set:{}'.format(tr_f[:5]))
data=pd.read_csv('/home/datta/Downloads/UrbanSound/urban-sound-classification/train/train.csv')
#data=pd.read_csv('train/train.csv')

print(data)

y=list(data['Class'].values)
#y=np.array(y)
#y=y.reshape(len(tr_f),1)




types=list(data.Class.unique())
print('Guessing from classes : {}'.format(types))




# In[45]:


count=0
print('Finding features....')
for file in tr_f:
	avg=[]
	x,sr=librosa.load('/home/datta/Downloads/UrbanSound/urban-sound-classification/train/Train/'+file)
#	x,sr=librosa.load('train/Train/'+file)

	#print(file,sr,np.shape(x))
	mfccs=librosa.feature.mfcc(x,sr=sr)
	#print(len(mfccs))
	if(count==0):
		a=[]
		for i in range(len(mfccs)):
			a.append(i)	
			b=pd.DataFrame(columns=a)
	#print(np.shape(mfccs))
	#print(np.shape(sum(mfccs)),len(sum(mfccs)))
	#print(mfccs[0],np.shape(mfccs[0]))	
	for i in range(len(mfccs)):
		avgapp=sum(mfccs[i])/len(mfccs[i])
		avg.append(avgapp)
		varapp=sum(np.power(mfccs[i]-avgapp,2))/len(mfccs[i])
		#avg.append(varapp)
#	print(mfccs,np.shape(mfccs),type(mfccs))
	avg=np.array(avg).reshape(1,len(avg))	
	#print(mfccs,np.shape(mfccs),type(mfccs))
	avg=pd.DataFrame(data=avg[0:],index=[int(file.replace('.wav',''))])	
	b=b.append(avg)
	count=count+1
	ll=int(len(tr_f)/6)	
	if(count%ll==0):
#	if(count%300==0):
		stop=timeit.default_timer()
		print('\nReached: {}'.format(count))
		print('time :{:.2f}'.format(stop-fstart))
print('Time taken for feature extraction: {:.2f}'.format(timeit.default_timer()-fstart))	
b=b.sort_index(axis=0)
p=pd.DataFrame(data=y,index=list(data['ID'].values),columns=['label'])
b=b.fillna(0)
b= pd.concat([b, p], axis=1)
b[a]=scaler.fit_transform(b[a])

X=b[a]


# In[53]:


b_tr,b_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.25,random_state=22)


# In[56]:



k=int(np.power(np.array(list((b_tr).shape)),0.5)[0])
if(k%2==0):
	k=k+1
print(k)
Rstart=timeit.default_timer()
scorek=0
for i in range(1,k+1):
	for j in range(2,3):

		knn = KNeighborsClassifier(n_neighbors = i,p=j)
		knn.fit(b_tr,y_tr)
		if(scorek < knn.score(b_ts,y_ts)):
			scorek = knn.score(b_ts,y_ts)
			finalk=[i,j]	
Rstop=timeit.default_timer()
print('Time taken for knn : {:.2f}'.format(Rstop-Rstart),'Score of knn: {:.2f}'.format(scorek),'[K,p] used ={}'.format(finalk))
knn = KNeighborsClassifier(n_neighbors = finalk[0],p=finalk[1])
knn.fit(b_tr,y_tr)
print('Classification Report for knn :- \n',classification_report(y_ts,knn.predict(b_ts)))


# In[163]:



from bokeh.models import Div
from bokeh.layouts import row, column,layout
from bokeh.plotting import figure, output_file, show, ColumnDataSource, curdoc
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc
test='5.wav'
print('Given input is: {}'.format(test))	
avg=[]
x,sr=librosa.load('/home/datta/Downloads/UrbanSound/urban-sound-classification/test/Test/'+test)
#print(file,sr,np.shape(x))
mfccs=librosa.feature.mfcc(x,sr=sr)
#print(len(mfccs)
a=[]
for i in range(len(mfccs)):
		a.append(i)	
		nb=pd.DataFrame(columns=a)
#print(np.shape(mfccs))
#print(np.shape(sum(mfccs)),len(sum(mfccs)))
#print(mfccs[0],np.shape(mfccs[0]))	
for i in range(len(mfccs)):
	avgapp=sum(mfccs[i])/len(mfccs[i])
	avg.append(avgapp)
	varapp=sum(np.power(mfccs[i]-avgapp,2))/len(mfccs[i])
	#avg.append(varapp)
#print(mfccs,np.shape(mfccs),type(mfccs))
avg=np.array(avg).reshape(1,len(avg))	
#print(mfccs,np.shape(mfccs),type(mfccs))
avg=pd.DataFrame(data=avg[0:],index=[int(test.replace('.wav',''))])	
nb=nb.append(avg)
f=nb[a]



model = Select(title="Model used to Predict", value="KNearestNeighbor",options=open(join(dirname(__file__),'models.txt')).read().split())
Plot = Select(title="Plot", value="Prediction",options=open(join(dirname(__file__),'plot.txt')).read().split())
Test = TextInput(title="Test file name",value="5.wav")
desc = Div(text=open(join(dirname(__file__), "description.html")).read())
Kn = Slider(start=1, end=k, value=5, step=1, title="Neighbours for KNN Prediction")
Kd = Slider(start=1, end=5, value=2, step=1, title="Distance Metric for KNN Prediction")
Rn = Slider(start=1, end=200, value=32, step=1, title="Numnber of Estimators for Random Forest Prediction")
Rmd = Slider(start=1, end=40, value=28, step=1, title="Max Depth for Random Forest Prediction")
Rmf = Slider(start=1, end=20, value=4, step=1, title="Max features for Random Forest Prediction")



source = ColumnDataSource(data=dict(x=[], y=[]))
plot = figure(y_range=(0, 2),plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
	


def call():

	print('Given input is: {}'.format(Test.value))	
	avg=[]
	x,sr=librosa.load('/home/datta/Downloads/UrbanSound/urban-sound-classification/test/Test/'+Test.value)
	#print(file,sr,np.shape(x))
	mfccs=librosa.feature.mfcc(x,sr=sr)
	#print(len(mfccs)
	a=[]
	for i in range(len(mfccs)):
			a.append(i)	
			nb=pd.DataFrame(columns=a)
	#print(np.shape(mfccs))
	#print(np.shape(sum(mfccs)),len(sum(mfccs)))
	#print(mfccs[0],np.shape(mfccs[0]))	
	for i in range(len(mfccs)):
		avgapp=sum(mfccs[i])/len(mfccs[i])
		avg.append(avgapp)
		varapp=sum(np.power(mfccs[i]-avgapp,2))/len(mfccs[i])
		#avg.append(varapp)
#	print(mfccs,np.shape(mfccs),type(mfccs))
	avg=np.array(avg).reshape(1,len(avg))	
	#print(mfccs,np.shape(mfccs),type(mfccs))
	avg=pd.DataFrame(data=avg[0:],index=[int(Test.value.replace('.wav',''))])	
	nb=nb.append(avg)
	f=nb[a]
	source.data = dict(x=x,y=y)
	update(f)
def update(f):
	if(Plot.value=="Prediction"):		
		if(model.value=="KNearestNeighbor"):
			clf = KNeighborsClassifier(n_neighbors=Kn.value,p=Kd.value)
		#n = Slider(start=1, end=k, value=5, step=1, title="Neighbours")		
		if(model.value=="RandomForest"):
			clf=RandomForestClassifier(n_estimators=Rn.value,max_depth=Rmd.value,max_features=Rmf.value)		
		if(model.value=="SupportVectorClassifier"):
			clf = SVC(random_state=0,tol=1e-5)		
			clf = CalibratedClassifierCV(clf)
		if(model.value=="LinearSVC"):
			clf = LinearSVC(random_state=0,tol=1e-5)
			clf = CalibratedClassifierCV(clf) 
		clf.fit(b_tr,y_tr)
	
		x=np.array(list(range(1,len(types)+1)))
		print(f)
		y=np.array((clf.predict_proba(f)[0]))
		print(y)
	if(Plot.value=="KNN-Score_vs_n"):
		x=range(1,k+1)
		y=[]			
		for n in x:
			clf=KNeighborsClassifier(n_neighbors=n,p=Kd.value)
			clf.fit(b_tr,y_tr)	
			y.append(clf.score(b_ts,y_ts))
	if(Plot.value=="KNN-Score_vs_p"):
		x=[1,2,3,4,5]
		y=[]			
		for n in x:
			clf=KNeighborsClassifier(n_neighbors=Kn.value,p=n)
			clf.fit(b_tr,y_tr)	
			y.append(clf.score(b_ts,y_ts))
	if(Plot.value=="RandomForest-Score_vs_n-estimators"):
		x=range(1,201)
		y=[]			
		for n in x:			
			clf=RandomForestClassifier(n_estimators=n,max_depth=Rmd.value,max_features=Rmf.value)
			clf.fit(b_tr,y_tr)	
			y.append(clf.score(b_ts,y_ts))
	if(Plot.value=="RandomForest-Score_vs_max-depth"):
		x=range(1,41)
		y=[]
		for n in x:			
			clf=RandomForestClassifier(n_estimators=Rn.value,max_depth=n,max_features=Rmf.value)
			clf.fit(b_tr,y_tr)	
			y.append(clf.score(b_ts,y_ts))
	if(Plot.value=="RandomForest-Score_vs_max-features"):
		x=range(1,20)
		y=[]
		for n in x:			
			clf=RandomForestClassifier(n_estimators=Rn.value,max_depth=Rmd.value,max_features=n)
			clf.fit(b_tr,y_tr)	
			y.append(clf.score(b_ts,y_ts))
	if(Plot.value=="FeatureImportance"):
		clf = KNeighborsClassifier(n_neighbors=Kn.value,p=Kd.value)
		n_feats = f.shape[1]
		y=[]
		for i in range(n_feats):
		    b_trs=b_tr.drop([i],axis=1)
		    b_tss=b_ts.drop([i],axis=1)    
		    clf.fit(b_trs,y_tr)
		    y.append(knn.score(b_tss,y_ts))
	source.data = dict(x=x,y=y)



controls=[Test,Plot,model,Kn,Kd,Rn,Rmd,Rmf]

for control in controls[1:len(controls)]:
	control.on_change('value', lambda attr, old, new: update(f))
Test.on_change('value', lambda attr, old, new: call())
inputs = column(*controls, width=320, height=1000)
inputs.sizing_mode = "fixed"
l = layout([[desc],
    	[inputs, plot],
	], sizing_mode="scale_both")
	

	#plot = figure(y_range=(0, 2),plot_width=400, plot_height=400)
	#plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

	#layout=column(plot,n)    
	#n.on_change('value', lambda attr, old, new: call())
call()
update(f)
curdoc().add_root(l)	
curdoc().title = "Model"

