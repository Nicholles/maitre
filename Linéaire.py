'''
Description:

####Format dataset########################################################################################
dataset=["taille","extension","annee_derniere_modif",label]


####Affichage dataset tensorflow##########################################################################
for elem in dataset:
	print(elem)
	
####Affichage de 4 lots######################################################################################
for batch in batch_dataset.take(4):
	print([arr.numpy() for arr in batch])

####Activer environnement virtuel#########################################################################
source ./venv/bin/activate
'''

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


####Variables Globales####################################################################################
nb_files=0
pimp = pathlib.Path('/home/nicholles/important_n')
imp = pathlib.Path('/home/nicholles/important')
dimensions=4
ext=np.array([])
transformer=OneHotEncoder(sparse=True)
scaler=MinMaxScaler()

###########################################################################################################
#Comptabilise le nombre total de fichiers
#Encode les extensions(prétraitement)
#Cree un tableau de la taille adéquate (Nombre de fichiers*(Nombre de caractéristiques+1<--dû au label))
###########################################################################################################
for item in list(pimp.glob('**/*')):
	if os.path.isfile(item):
		nb_files+=1
		ext=np.append(ext,os.path.splitext(item)[1])
for item in list(imp.glob('**/*')):
	if os.path.isfile(item):
		nb_files+=1
		ext=np.append(ext,os.path.splitext(item)[1])	
		
dataset=np.empty([nb_files,dimensions],dtype=float)
transformer.fit(ext.reshape(-1, 1))
ext=transformer.transform(ext.reshape(-1, 1))


############################################################################################################
#Création d'un dataset tensorflow, avec nos 3 critères et notre label qu'on normalise et mélange aleatoirement(prétraitement)
############################################################################################################
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
tf_dataset =tf_dataset.shuffle(nb_files)

############################################################################################################
#Séparation dataset(train et test) et création de groupes de fichiers
############################################################################################################
TRAIN_SIZE = int(0.8 * nb_files)
TEST_SIZE = int(0.2 * nb_files)
BATCH_SIZE = int(nb_files/15)		
train_dataset = tf_dataset.take(TRAIN_SIZE)
test_dataset= tf_dataset.skip(TRAIN_SIZE)
train_dataset =train_dataset.shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
test_dataset =test_dataset.batch(BATCH_SIZE)



############################################################################################################
#Modèle1
############################################################################################################
model1 = tf.keras.Sequential([
		tf.keras.experimental.LinearModel()
		])

	
############################################################################################################
#Entrainement du modèle
############################################################################################################
print('#####################################################################################################')	
print('ENTRAINEMENT')
print('#####################################################################################################')
model1.compile(optimizer='Adam', loss=tf.keras.losses.Hinge(),metrics=['acc'])
model1.fit(train_dataset,epochs=100)

############################################################################################################
#évaluation du modèle
############################################################################################################
print('#####################################################################################################')	
print('EVALUATION')
print('#####################################################################################################')
model1.evaluate(test_dataset)


iterator = test_dataset.__iter__() 
for i in range(4):
	x_batch , y_batch = iterator.get_next()
	for j in range(len(y_batch)):
		if model1.predict(x_batch)[j]<0.5:
			print('Label prédit:0','label réel:',y_batch[j].numpy())
		else :
			print('Label prédit:1','label réel:',y_batch[j].numpy())
		

