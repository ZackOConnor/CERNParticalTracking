import trackml
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import glob

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

train = np.unique([p.split('-')[0] for p in sorted(glob.glob('./train_data/**.csv'))])
test = np.unique([p.split('-')[0] for p in sorted(glob.glob('./test_data/**.csv'))])
sub = np.unique([p.split('-')[0] for p in sorted(glob.glob('./sample_submission/**.csv'))])

model = Sequential() 
model.add(Embedding(50000,100, input_length=11))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
# Creating the Model

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Defines loss function and prediction metric

# the +10000 shifts the event space into a non-negitive space, the model won't take negitives
shift=10000
for e in train:
    hits, cells, truth = load_event(e, parts=['hits', 'cells', 'truth'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(['mean']).reset_index()
    cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]
    hits = pd.merge(hits, truth, how='left', on = 'hit_id')
    # Pulls in all needed data from the diffrent folders on a event by event bases
    print(e,"Train")
    hits['x'] = hits['x']+shift
    hits['y'] = hits['y']+shift
    hits['z'] = hits['z']+shift
    hits['particle_id'] = model.fit(hits[['hit_id','x','y','z','volume_id','layer_id','module_id','event_id','ch0','ch1','value']].values,hits['particle_id'],batch_size=10000, epochs = 1)
    # Fits the model
    # Batch Size and epoch were choosen with shortest training time as #1 
# Traning

df_test = []

for e in test:
    hits, cells = load_event(e, parts=['hits', 'cells'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(['mean']).reset_index()
    cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]
    hits = pd.merge(hits, truth, how='left', on = 'hit_id')
    # Pulls in all needed data from the diffrent folders on a event by even bases
    print(e,"Test")
    hits['x'] = hits['x']+shift
    hits['y'] = hits['y']+shift
    hits['z'] = hits['z']+shift
    # the +10000 shifts the event space into a non-negitive space, the model won't take negitives
    hits['particle_id'] = model.predict(hits[['hit_id','x','y','z','volume_id','layer_id','module_id','event_id','ch0','ch1','value']].values,verbose=1)
    # Predicts the test set
    df_test.append(hits[['event_id','hit_id','particle_id']].copy())
# Testing

df_test = pd.concat(df_test, ignore_index=True)
sub = pd.merge(sub, df_test, how='left', on=['event_id','hit_id']) #for submission
sub['track_id'] = sub['particle_id'] + 1
sub['track_id'] = sub['track_id'].astype(int)#for submission
sub[['event_id','hit_id','track_id']].to_csv('submission-001.csv', index=False) #for submission
#creates the submission file by combining the event_id and partical_id on hit_id  