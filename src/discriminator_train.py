import pickle
import numpy as np
# Disable GPU if model uses LSTM instead of GPU-optimized CuDNNLSTM
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, TimeDistributed, \
    CuDNNLSTM
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


# Define RNN model for 10-dimensional data of unknown length; three layers; 32-16-2
def discriminator(dropout=0.2):
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, 10), return_sequences=True))
    # model.add(CuDNNLSTM(32, input_shape=(None, 10), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(16, return_sequences=False))
    # model.add(CuDNNLSTM(16, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Read in cleaned/encoded data from pickle file
pickle_in = open('cleanData', "rb")
realMatches = pickle.load(pickle_in)  # real matched sequence pairs
fakeMatches = pickle.load(pickle_in)  # fake matched sequence pairs
X = pickle.load(pickle_in)  # encoded matched sequence pairs
Y = pickle.load(pickle_in)  # associated label: 1 for real, 0 for fake
pickle_in.close()

# Initialize model
model = discriminator()

# Print model summary
model.summary()

# Set ratio for portion of data set reserved for training
trainRatio = 0.9

# Set number of iterations for validation set
numVal = int(len(X) * (1 - trainRatio))


# Split the encoded data into validation set and training set
# Last numVal elements will be for validation, rest of them are for training


# Validation set generator on which losses will be evaluated and metrics given
def validation_generator():
    i = -1
    m = numVal
    while True:
        i = (i + 1) % m
        yield X[-numVal + i], Y[-numVal + i]


# Training set generator that composes single batch;
def train_generator():
    i = -1
    m = len(X) - numVal
    while True:
        i = (i + 1) % m
        yield X[i], Y[i]


# Collect logging data during training
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

# Stop if no improvement in validation_loss between epochs
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, restore_best_weights=True)

# Save each epoch's model separately
modelcheckpoint = ModelCheckpoint("discriminator-improvement--{epoch:02d}--{val_loss:.2f}.h5", monitor='val_loss',
                                  save_best_only=True, mode='auto', verbose=0)

# Train the model with provided data sets
stepsPerEpoch = 100  # batch size
numEpochs = int(4 * (len(X) - numVal) / (stepsPerEpoch * 5))  # keep numEpochs reasonable, i.e. < 1000

historyObject = model.fit_generator(train_generator(), validation_data=validation_generator(),
                                    steps_per_epoch=stepsPerEpoch, epochs=numEpochs, verbose=1,
                                    validation_steps=numVal, callbacks=[tensorboard, earlystopping, modelcheckpoint],
                                    workers=8)

# Organize history of discriminator training for later plotting
val_loss = np.array(historyObject.history['val_loss'])
val_acc = np.array(historyObject.history['val_accuracy'])
loss = np.array(historyObject.history['loss'])
acc = np.array(historyObject.history['accuracy'])

hist = np.vstack((val_loss, val_acc, loss, acc)).T

# Save the organized history into a txt file
np.savetxt(str(trainRatio * 100) + "_history.txt", hist, delimiter=",")

# Save the fitted model as an h5 file
# Contains model architecture, weights, training configuration (loss, optimizer), state of optimizer
model.save('discriminator_model.h5')
