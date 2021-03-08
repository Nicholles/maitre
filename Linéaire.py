#source ./venv/bin/activate  # sh, bash, or zsh

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import os,time
import tensorflow as tf
import shutil
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


np.set_printoptions(precision=4)


#Variables####################################################
num_points=0
num_test=0
pimp = pathlib.Path('/home/nicholles/0')
imp = pathlib.Path('/home/nicholles/1')
dimensions=4
ext=np.array([])
transformer=OneHotEncoder(sparse=True)
scaler=MinMaxScaler()

#Connnaitre le nombre de fichiers et adapter les extensions####################################################
for item in list(pimp.glob('**/*')):
	if os.path.isfile(item):
		num_points+=1
		ext=np.append(ext,os.path.splitext(item)[1])
		
for item in list(imp.glob('**/*')):
	if os.path.isfile(item):
		num_points+=1
		ext=np.append(ext,os.path.splitext(item)[1])

		
dataset=np.empty([num_points,dimensions],dtype=float)
transformer.fit(ext.reshape(-1, 1))
ext=transformer.transform(ext.reshape(-1, 1))
#print(ext[:,0])

#Création et prétraitement dataset####################################################

cmpt=0
for item in list(pimp.glob('**/*')):
	if os.path.isfile(item):
		dataset[cmpt,:]=[os.path.getsize(item),ext[cmpt,0],pd.to_datetime(time.ctime(os.stat (item).st_mtime)).year,0]
		cmpt+=1
for item in list(imp.glob('**/*')):
	if os.path.isfile(item):
		dataset[cmpt,:]=[os.path.getsize(item),ext[cmpt,0],pd.to_datetime(time.ctime(os.stat (item).st_mtime)).year,1]
		cmpt+=1



scaler.fit(dataset[:,0].reshape(-1, 1))
dataset[:,0]=((scaler.transform(dataset[:,0].reshape(-1, 1))).reshape(1, -1))*10
scaler.fit(dataset[:,2].reshape(-1, 1))
dataset[:,2]=((scaler.transform(dataset[:,2].reshape(-1, 1))).reshape(1, -1))*10
tf_dataset = tf.data.Dataset.from_tensor_slices((dataset[:,:3],dataset[:,3]))
tf_dataset =tf_dataset.shuffle(num_points)


#Séparation dataset(train et test)####################################################
TRAIN_SIZE = int(0.8 * num_points)
TEST_SIZE = int(0.2 * num_points)
BATCH_SIZE = int(num_points/15)
		
train_dataset = tf_dataset.take(TRAIN_SIZE)
test_dataset= tf_dataset.skip(TRAIN_SIZE)
train_dataset =train_dataset.shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
test_dataset =test_dataset.batch(BATCH_SIZE)




'''print('##########################################################################################')
for elem in train_dataset:
	print(elem)
print('##########################################################################################')	
for elem in test_dataset:
	print(elem)
print('##########################################################################################')

'''

#Initilisation et entrainement model/estimateur##########################################
print('##########################################################################################')	
print('Entrainement')
print('##########################################################################################')
model = tf.keras.Sequential([
	tf.keras.experimental.LinearModel(),
	tf.keras.layers.Dense(2)
])
	
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])
model.fit(train_dataset,epochs=100)

#evaluate model/estimateur##########################################
print('##########################################################################################')	
print('evaluation')
print('##########################################################################################')

model.evaluate(test_dataset)
'''for elem in test_dataset.take(4):
	print(elem)'''
prediction=model.predict(test_dataset)
#print(prediction)
#print(np.argmax(prediction[0]),np.argmax(prediction[1]),np.argmax(prediction[2]),np.argmax(prediction[3]))
for i in prediction:
	print(np.argmax(i))



