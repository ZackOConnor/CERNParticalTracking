from trackml.dataset import load_event
import pandas as pd

def test(model, test, shift=10000):
    """create prediction from train model over the test data. then append the prediction to the test data tp create the submission
    model   -- trained model
    test    -- test data
    shift   -- coordinate shift needed to eliminate negitive numbers
    """
    df_test = []

    for e in test:
        hits, cells = load_event(e, parts=['hits', 'cells'])
        hits['event_id'] = int(e[-9:])
        cells = cells.groupby(by=['hit_id'])[
            'ch0', 'ch1', 'value'].agg(['mean']).reset_index()
        cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
        hits = pd.merge(hits, cells, how='left', on='hit_id')
        # Pulls in all needed data from the diffrent folders on a event by even bases
        hits = pd.merge(hits, truth, how='left', on='hit_id')
        print(e, "Test")
        # the +10000 shifts the event space into a non-negitive space, the model won't take negitives
        hits['x'] += shift
        hits['y'] += shift
        hits['z'] += shift
        # Predicts the test set
        hits['particle_id'] = model.predict(
            hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id', 'event_id', 'ch0', 'ch1', 'value']].values, verbose=1)
        # append the predictions onto the test data frame
        df_test.append(hits[['event_id', 'hit_id', 'particle_id']].copy())

    return df_test
