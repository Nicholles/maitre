'''
Description:

####Format############################################
dataset=["taille","extension",
	"jour_crea","mois_crea","annee_crea",
	"jour_modif","mois_modif","annee_modif",
	"jour_acces","mois_acces","annee_acces",
	"categ_taille","categ_ext","categ_date_crea","categ_date_modif", "categ_date_acces",
	"duree_modif_acces","duree_modif_crea"]

categ_taille:vide=0Ko;tres petit=0-10Ko ;petit=10-100Ko;moyen=100-1Mo ;grand=1Mo-16Mo ;tres grand=16-128Ko ;gigantesque>128Ko*******peut etre pas meilleure idée

categ_ext:audio; compressés; exécutables; images; internet; langage de prog; office; videos; vindows; dessin_archi; 

categ_date_:ancien>5,+_recent=5-2ans,recent<2ans

####Affichage datasets generateur python############################################
for l in obt():
	print(str(l))	

####Affichage dataset tensorflow############################################
for elem in dataset:
	print(elem)
	tf.compat.v1.train.limit_epochs(elem, num_epochs=1)	
	
####Affichage de lot############################################
for batch in batch_dataset.take(4):
	print([arr.numpy() for arr in batch])
	

'''


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import os,time
import tensorflow as tf

tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()
import seaborn as sns

np.set_printoptions(precision=4)


#creation path
p = pathlib.Path('.')
dimensions=7


#Donne le nombre de données####################################################
num_points=0
for item in list(p.glob('**/*')):
	if os.path.isfile(item):
		num_points+=1
	
		
points=np.empty([num_points,dimensions],dtype=float)


#générateur python####################################################
def obt():
	cmpt=0
	for item in list(p.glob('**/*')):
		if os.path.isfile(item):
			points[cmpt,0]=os.path.getsize(item)
			points[cmpt,1]=pd.to_datetime(time.ctime(os.stat (item).st_mtime)).day
			points[cmpt,2]=pd.to_datetime(time.ctime(os.stat (item).st_mtime)).month
			points[cmpt,3]=pd.to_datetime(time.ctime(os.stat (item).st_mtime)).year
			points[cmpt,4]=pd.to_datetime(time.ctime(os.stat (item).st_atime)).day
			points[cmpt,5]=pd.to_datetime(time.ctime(os.stat (item).st_atime)).month
			points[cmpt,6]=pd.to_datetime(time.ctime(os.stat (item).st_atime)).year
			cmpt+=1
	yield points
		

#Creation dataset tensorflow##Pipeline:Transformer+estimateur###########




#Pretraitement des données


def input_fn():
	dataset =tf.data.Dataset.from_generator(obt,output_signature=(tf.TensorSpec(shape=(num_points,7),dtype=tf.float32)))
	return dataset

#Melange aleatoire des donnees/creation de lots/repetition du dataset



#Initilisation model/estimateur
num_clusters = 2
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)


#Entraînez le modèle
num_iterations = 10
previous_centers = None

for _ in range(num_iterations):
	kmeans.train(input_fn)
	cluster_centers = kmeans.cluster_centers()
	if previous_centers is not None:
		print( 'delta:', cluster_centers - previous_centers)
	previous_centers = cluster_centers
	print ('score:', kmeans.score(input_fn))
	
print ('cluster centers:', cluster_centers)


# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
	cluster_index = cluster_indices[i]
	center = cluster_centers[cluster_index]
	print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)









