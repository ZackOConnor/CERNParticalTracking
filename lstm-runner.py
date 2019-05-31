from data_loader import load_data
from model import make_model
from train import train
from test import test
from submission import make_submission

#loads in train, test, and sub data
data_train, data_test, data_sub = load_data()

#pulls in and trains the model
model = make_model()
model = train(model, data_train)

#runs the test data through the model
df_test = test(model, data_test)

#makes the sumbission form the labeled test data
make_submission(df_test, data_sub)
