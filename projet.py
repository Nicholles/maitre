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

categ_date_:ancien>5,+_recent=5-2ans,recent<2ans

####Affichage datasets generateur python############################################
for m,i,j,k,l in obt():
	print(m, ":",i, ":", str(j),":", str(k),":", str(l))	

####Affichage dataset tensorflow############################################
for elem in dataset.take(4):
	print(elem)	
	
####Affichage de lot############################################
for batch in batch_dataset.take(4):
	print([arr.numpy() for arr in batch])
	
Mes questionnements:
il y a des fichiers sans extensions
données déséquilibrées?
trouver le nombre de données max utile
faire des calculs sur "duree_modif_acces"?
obsolescence des extensions
categ_ext:audio; compressés; exécutables; images; internet; langage de prog; office; videos; vindows; dessin_archi; 
inutile=.temp, .bak, .wbk, .pif, .diz, .chk, .gid, .bad, .old; inutile_pos=.flv, .avi, .mov, .mpg, .mp4,.asf, .wmv, .wma, .nut, .rm,ogg, .ogv, .oga, .ogx, .spx, .opus, .ogm,.vob, .ifo ; autre
pd.to_datetime(
    raw_interventions['Date effective'], dayfirst=True)

'''

#source ./venv/bin/activate  # sh, bash, or zsh
#python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import os,time
import tensorflow as tf
import seaborn as sns

np.set_printoptions(precision=4)

'''
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
'''


#creation path
p = pathlib.Path('.')


#générateur python####################################################
def obt():
	
	for item in list(p.glob('**/*')):
		if os.path.isfile(item):
			
			yield os.path.getsize(item), os.path.splitext(item)[1], pd.to_datetime(time.ctime(os.stat (item).st_mtime)).day, pd.to_datetime(time.ctime(os.stat (item).st_mtime)).month, pd.to_datetime(time.ctime(os.stat (item).st_mtime)).year, pd.to_datetime(time.ctime(os.stat (item).st_atime)).day,  pd.to_datetime(time.ctime(os.stat (item).st_atime)).month, pd.to_datetime(time.ctime(os.stat (item).st_atime)).year

	
cmpt=0
for item in list(p.glob('**/*')):
	if os.path.isfile(item):
		cmpt+=1	

#Creation dataset tensorflow##Pipeline:Transformer+estimateur###########
dataset = tf.data.Dataset.from_generator(obt, output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.string),
tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),
tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),
tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32)) )
for elem in dataset.take(2):
	print(elem)


#Pretraitement des données
TRAIN_SIZE = int(0.8 * cmpt)
TEST_SIZE = int(0.2 * cmpt)
BUFFER_SIZE = int(cmpt)
BATCH_SIZE = int(cmpt/4)
REPEAT_SIZE = 2
dataset =dataset.shuffle(BUFFER_SIZE)
train_dataset = dataset.take(TRAIN_SIZE)
test_dataset = dataset.skip(TRAIN_SIZE)
#Melange aleatoire des donnees/creation de lots/repetition du dataset
train_dataset =train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True).repeat(REPEAT_SIZE).prefetch(2)
print(dataset)
print(train_dataset)
print(test_dataset)
for elem in test_dataset.take(2):
	print(elem)

kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

'''#Initilisation model/estimateur

class Clustering(tf.keras.Model):
  def __init__(self):
    super(Clustering, self).__init__()
    self.cluster =  tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)

    )

  def call(self, x):
    cluterised = self.cluster(x)
    return decoded

clustering = Clustering()

clustering.compile(optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
     run_eagerly=None, steps_per_execution=None
)


#Entraînez le modèle
clustering.fit(
    train_dataset, y=None, epochs=1, callbacks=None,
   shuffle=True, steps_per_epoch=None,
    validation_steps=None, validation_freq=1
)


#Evaluer le modèle

clustering.evaluate(
    x=None, y=None, steps=None, callbacks=None, return_dict=False
)



#Utiliser le modèle
clustering.predict(
    train_dataset, steps=None, callbacks=None
)



#Point de controle de l'iterateur
def pt_cont(dataset):
	new_model = tf.keras.Model(...)
	checkpoint = tf.train.Checkpoint(model=new_model)
	#manager = tf.train.CheckpointManager(ckpt, '/tmp/my_ckpt', max_to_keep=3)
	save_path = checkpoint.save('/tmp/training_checkpoints')
	checkpoint.restore(save_path)'''








