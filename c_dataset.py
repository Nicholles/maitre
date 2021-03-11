'''
Description:

####Format dataset########################################################################################
dataset=["taille","extension","annee_derniere_modif"]


####Affichage datasets generateur python##################################################################
for l in obt():
	print(str(l))	

####Affichage dataset tensorflow##########################################################################
for elem in dataset:
	print(elem)
	
####Affichage en lot######################################################################################
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

from sklearn.preprocessing import MinMaxScaler #utilisé pour normaliser
from sklearn.preprocessing import OneHotEncoder #utilisé pour le traitement des features


np.set_printoptions(precision=4) #precision de notre affichage


####Variables Globales####################################################################################
p = pathlib.Path('/home/nicholles/folder') #Dossier où se trouve nos fichiers sans label
dimensions=3 				    #Nombre de caractéristiques, du fichier, utilisées
transformer=OneHotEncoder(sparse=True) 
scaler=MinMaxScaler()
nb_files=0				    #Nombre de fichier à labeliser
name=[]				    
ext=np.array([])

###########################################################################################################
#Renvoie:
#Le nombre total de fichiers à labiliser
#Le nom de ses fichiers
#Leurs extensions prétraitées
#Un tableau de la taille adéquate (Nombre de fichiers*Nombre de caractéristiques)
###########################################################################################################
def fn(nb_files,name,ext):
	for item in list(p.glob('**/*')):
		if os.path.isfile(item):
			nb_files+=1
			name.append(str(item))
			ext=np.append(ext,os.path.splitext(item)[1])
	transformer.fit(ext.reshape(-1, 1))
	ext=transformer.transform(ext.reshape(-1, 1))
	return nb_files,name,ext
				
############################################################################################################
#Générateur python qui renvoie les 3 caractéristiques sur lesquelles on va s'appuyer pour labéliser les fichiers
############################################################################################################
def obt():
	cmpt=0
	for item in list(p.glob('**/*')):
		if os.path.isfile(item):
			donnees[cmpt,0]=os.path.getsize(item)
			donnees[cmpt,1]=ext[cmpt,0]
			donnees[cmpt,2]=pd.to_datetime(time.ctime(os.stat (item).st_mtime)).year
			cmpt+=1
	scaler.fit(donnees[:,0].reshape(-1, 1))
	donnees[:,0]=((scaler.transform(donnees[:,0].reshape(-1, 1))).reshape(1, -1))*10
	
	scaler.fit(donnees[:,2].reshape(-1, 1))
	donnees[:,2]=((scaler.transform(donnees[:,2].reshape(-1, 1))).reshape(1, -1))*10
	yield donnees
		


############################################################################################################
#Création d'un dataset tensorflow, avec nos 3 critères, qu'on mélange aleatoirement
############################################################################################################
nb_files,name,ext=fn(nb_files,name,ext)
donnees=np.empty([nb_files,dimensions],dtype=float)

def input_fn():
	dataset=tf.data.Dataset.from_generator(obt, output_signature=(tf.TensorSpec(shape=(nb_files,dimensions),dtype=tf.float32)))
	train_dataset =dataset.shuffle(nb_files)
	return train_dataset


############################################################################################################
#Initilisation Kmeans
############################################################################################################
num_clusters = 2
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

############################################################################################################
#Fait tourner l'algorithme
############################################################################################################
num_iterations = 5
previous_centers = None


for _ in range(num_iterations):
	kmeans.train(input_fn)
	cluster_centers = kmeans.cluster_centers()
	if previous_centers is not None:
		print( 'delta:', cluster_centers - previous_centers)
	previous_centers = cluster_centers
print ('cluster centers:', cluster_centers)

############################################################################################################
# Association les donnees à leurs clusters et creation 2 dossiers l'un avec les fichiers importants l'autre avec les fichiers pas importants
############################################################################################################
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, file in enumerate(donnees):
	cluster_index = cluster_indices[i]
	center = cluster_centers[cluster_index]
	#print ('Le fichier:', file, 'est dans le cluster', cluster_index, 'centré e', center)
	os.makedirs('/home/nicholles/important', exist_ok=True)
	os.makedirs('/home/nicholles/important_n', exist_ok=True)
	if cluster_centers[0,2]>cluster_centers[1,2]:
		if cluster_index==0:
			shutil.copy2(name[i],'/home/nicholles/important')
		else:
			shutil.copy2(name[i],'/home/nicholles/important_n')	
	else:
		if cluster_index==0:
			shutil.copy2(name[i],'/home/nicholles/important_n')
		else:
			shutil.copy2(name[i],'/home/nicholles/important')

############################################################################################################
# Mettre tous les executables et fichiers binaires dans les fichiers importants
############################################################################################################
pimp = pathlib.Path('/home/nicholles/important_n')
for item in list(pimp.glob('**/*')):
	if os.path.isfile(item):
		if os.path.splitext(item)[1]=='.exe' or os.path.splitext(item)[1]=='':
			shutil.copy2(str(item),'/home/nicholles/important')
			os.remove(str(item))

