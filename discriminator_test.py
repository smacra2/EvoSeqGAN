import pickle
# Disable GPU if model uses LSTM instead of GPU-optimized CuDNNLSTM
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
from keras.models import load_model
from keras.utils import plot_model

# Read in cleaned/encoded data from pickle file
pickle_in = open('cleanData', "rb")
realMatches = pickle.load(pickle_in)  # real matched sequence pairs
fakeMatches = pickle.load(pickle_in)  # fake matched sequence pairs
X = pickle.load(pickle_in)  # encoded matched sequence pairs
Y = pickle.load(pickle_in)  # associated label: 1 for real, 0 for fake
pickle_in.close()

# Load the fitted model
model = load_model('discriminator_model.h5')

# Print model summary and plot it
# model.summary()
# plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

# Test the model on arbitrary inputs
# Example reference seq: CTG, candidate seq: CGG
# Note length is 3; thus create (1,3,10) dimensional matrix
# Initial 1 creates first dimension, 3 is sequence length, 10 is dimension of the data.
# Recall data encoding: refA: 1000000000, refC: 0100000000, refG: 0010000000, refT: 0001000000, ref-: 0000100000
# cndA: 0000010000, cndC: 0000001000, cndG: 0000000100, cndT: 0000000010, cnd-: 0000000001
# i.e. reference seq takes dimensions 0-4, candidate seq takes dimensions 5-9
test = np.zeros((1, 3, 10))

test[0, 0, 1] = 1  # reference's C
test[0, 0, 6] = 1  # candidate's C
test[0, 1, 3] = 1  # reference's T
test[0, 1, 6] = 1  # candidate's G
test[0, 2, 2] = 1  # reference's G
test[0, 2, 7] = 1  # candidate's G

yhat = model.predict(test)
if yhat[0][1] < 0.5:
    print("Fake with score: " + str(yhat[0][1]))
else:
    print("Real with score: " + str(yhat[0][1]))

print()

# Set number of test iterations
numTest = 2000

totalCorrect = 0
count = 0
for i in range(len(X) - numTest, len(X)):
    yhat = model.predict(X[i])
    # totalCorrect++ if model prediction is correct
    if yhat[0][Y[i][0]] >= 0.5:
        totalCorrect += 1
    count += 1
    print("Sequence number: " + str(i) + " has actual score: " + str(Y[i][0]) + " and predicted score: " +
          str(yhat[0][1]))

print("Model accuracy:" + str(totalCorrect * 1.0 / count))
