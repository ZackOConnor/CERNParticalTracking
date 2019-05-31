from data_loader import load_data
from model import make_model
from train import train
from test import test
from submission import make_submission


data_train, data_test, data_sub = load_data()
#loads in train, test, and sub data

model = make_model()
model = train(model, data_train)
#pulls in and trains the model

df_test = test(model, data_test)
#runs the test data through the model

make_submission(df_test, data_sub)
#makes the sumbission form the labeled test data