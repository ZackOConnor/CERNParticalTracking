import pandas as pd


def make_submission(df_test, sub):
    """Merge the event_id and hit_id back on to the test with prediction data frame to create the submission data frame
        params:
        df_test -- test data frame with prediction apended on
        sub     -- the submission file with out predictions
    """
    df_test = pd.concat(df_test, ignore_index=True)
    #merge test with prediction data frame onto the submissions data frame
    sub = pd.merge(sub, df_test, how='left', on=[
                   'event_id', 'hit_id'])
    sub['track_id'] = sub['particle_id'] + 1
    sub['track_id'] = sub['track_id'].astype(int)
    sub[['event_id', 'hit_id', 'track_id']].to_csv(
    'submission-001.csv', index=False)
    #merge the test data onto the submission data frame and then create a csv
