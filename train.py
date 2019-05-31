from trackml.dataset import load_event
import pandas as pd
import numpy as np

def train(model, train, shift=10000, polar=0):
    for e in train:
        hits, cells, truth = load_event(e, parts=['hits', 'cells', 'truth'])
        hits['event_id'] = int(e[-9:])
        cells = cells.groupby(by=['hit_id'])['ch0', 'ch1','value'].agg(['mean']).reset_index()
        cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
        hits = pd.merge(hits, cells, how='left', on='hit_id')
        # col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]
        hits = pd.merge(hits, truth, how='left', on='hit_id')
        print(e, "Train")
        hits['x'] = hits['x'] + shift
        hits['y'] = hits['y'] + shift
        hits['z'] = hits['z'] + shift
        if polar == 1:
            hits['radius'] = np.sqrt(hits['y']*hits['y']+hits['x']*hits['x'])
            hits['theta'] = np.tan(hits['y']/hits['x'])**-1
            hits['particle_id'] = model.fit(hits[['hit_id', 'radius', 'theta', 'z', 'volume_id', 'layer_id', 'module_id',
                                              'event_id', 'ch0', 'ch1', 'value']].values, hits['particle_id'], batch_size=10000, epochs=1)
        else:
            hits['particle_id'] = model.fit(hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id',
                                              'event_id', 'ch0', 'ch1', 'value']].values, hits['particle_id'], batch_size=10000, epochs=1)
    return model