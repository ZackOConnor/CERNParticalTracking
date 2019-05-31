import pandas as pd


def make_submission(df_test, sub):
    df_test = pd.concat(df_test, ignore_index=True)
    sub = pd.merge(sub, df_test, how='left', on=[
                   'event_id', 'hit_id'])
    sub['track_id'] = sub['particle_id'] + 1
    sub['track_id'] = sub['track_id'].astype(int)
    sub[['event_id', 'hit_id', 'track_id']].to_csv(
    'submission-001.csv', index=False)
    #merge the test data onto the submission data frame and then create a csv