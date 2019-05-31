from trackml.dataset import load_event
import pandas as pd
import numpy as np

def train(model, train, shift=10000, polar=0):
    """Takes in the training data, cleans it, and the uses it to train the model
    params
    model -- created, but untrained model
    train -- training data set
    shift -- coordinate shift needed to eliminate negative numbers
    polar -- used to turn polar coordinate on and off
    """
    for e in train:
        #seprates out the hits, cells, and truth from the data
        hits, cells, truth = load_event(e, parts=['hits', 'cells', 'truth'])
        hits['event_id'] = int(e[-9:])
        #group the data by hit ID sorted by channel and mean value
        cells = cells.groupby(by=['hit_id'])['ch0', 'ch1','value'].agg(['mean']).reset_index()
        cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
        hits = pd.merge(hits, cells, how='left', on='hit_id')
        hits = pd.merge(hits, truth, how='left', on='hit_id')
        print(e, "Train")
        #applies the shift to eliminate negitive numbers
        hits['x'] += shift
        hits['y'] += shift
        hits['z'] += shift
        #If polar coordinate are turned on, calculate the radious and theta and use them to train the model
        if polar == 1:
            hits['radius'] = np.sqrt(hits['y']*hits['y']+hits['x']*hits['x'])
            hits['theta'] = np.tan(hits['y']/hits['x'])**-1
            hits['particle_id'] = model.fit(hits[['hit_id', 'radius', 'theta', 'z', 'volume_id', 'layer_id', 'module_id',
                                              'event_id', 'ch0', 'ch1', 'value']].values, hits['particle_id'], batch_size=10000, epochs=1)
        else:
        #Train the model on the non polar x, y, and z values
            hits['particle_id'] = model.fit(hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id',
                                              'event_id', 'ch0', 'ch1', 'value']].values, hits['particle_id'], batch_size=10000, epochs=1)
    return model
