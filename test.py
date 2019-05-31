from trackml.dataset import load_event
import pandas as pd


def test(model, test, shift=10000):
    df_test = []

    for e in test:
        hits, cells = load_event(e, parts=['hits', 'cells'])
        hits['event_id'] = int(e[-9:])
        cells = cells.groupby(by=['hit_id'])[
            'ch0', 'ch1', 'value'].agg(['mean']).reset_index()
        cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
        hits = pd.merge(hits, cells, how='left', on='hit_id')
        # col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]
        hits = pd.merge(hits, truth, how='left', on='hit_id')
        # Pulls in all needed data from the diffrent folders on a event by even bases
        print(e, "Test")
        hits['x'] = hits['x'] + shift
        hits['y'] = hits['y'] + shift
        hits['z'] = hits['z'] + shift
        # the +10000 shifts the event space into a non-negitive space, the model won't take negitives
        hits['particle_id'] = model.predict(
            hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id', 'event_id', 'ch0', 'ch1', 'value']].values, verbose=1)
        # Predicts the test set
        df_test.append(hits[['event_id', 'hit_id', 'particle_id']].copy())
        # append the predictions onto the test data frame

    return df_test