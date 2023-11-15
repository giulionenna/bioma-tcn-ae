# %%
import numpy
import time
from utilities import select_gpus, plot_results_swat# utilities.py: Contains a few miscellaneous functions 
from tcnae import TCNAE # tcnae.py: Specification of the TCN-AE model
import data_swat # data.py: Allows to generate anomalous Mackey-Glass (MG) time series 
import argparse

parser = argparse.ArgumentParser(description="Train the model on the swat dataset")
parser.add_argument("--window_length", type=int, default=50)
parser.add_argument("--window_stride",type=int, default=25)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()



# If you have several GPUs, select one or more here (in a list)
#select_gpus(0)

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# %%
train_ts_id = 1 # [1-10]. Train the model on Mackey-Glass time series 1
data_gen = data_swat.DataSwat(window_length = args.window_length, ratio = 1, window_stride=args.window_stride, error_window_length=25)
train_data = data_gen.build_data() # Returns a dictionary
train_X = train_data["train_X"] # We only need train_X (input = output) for the training process
print("train_X.shape:", train_X.shape) # A lot of training sequences of length 1050 and dimension 1

# %%
import keras.backend as K
K.clear_session()

# %%

# Build and compile the model
#
tcn_ae = TCNAE(ts_dimension=train_X.shape[2], dilations=(1, 2, 4, 8, 16, 32, 64), latent_sample_rate=args.window_length, nb_filters=50) # Use the parameters specified in the paper

#
# Train TCN-AE for 10 epochs. For a better accuracy 
# on the test case, increase the epochs to epochs=40 
# The training takes about 3-4 minutes for 10 epochs, 
# and 15 minutes for 40 epochs (on Google CoLab, with GPU enabled)
#
tcn_ae.fit(train_X, train_X, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

# %%


# %%
#
# Test the model on another Mackey-Glass time series
# Might take a few minutes...
#
start_time = time.time()

#
# Take the whole time series... Like the training data, the test data is standardized (zero mean and unit variance)
#
test_X = train_data['test_X'] 
test_labels = train_data['test_labels']# We need an extra dimension for the batch-dimension
print("test_X.shape", test_X.shape) # This is one long time series
anomaly_score = tcn_ae.predict_cosine(test_X)
print("> Time:", round(time.time() - start_time), "seconds.")

# %%
 

# %%
#
# Make a plot of the anomaly-score and see how it matches the real anomaly windows
# Vertical red bars show the actual anomalies.
# Vertical yellow bars show regions which can be ignored (usually start and 
# end of a time series, which lead to transient behavior for some algorithms).
# The blue curve is the anomaly score.
# The red horizontal line indicates a simple threshold, which is the smallest possible value that would not produce a false positive

#
from sklearn.metrics import roc_auc_score
plot_results_swat(test_X, test_labels, anomaly_score, pl_range = None, plot_signal = False, plot_anomaly_score = True)
auc_roc = roc_auc_score((test_labels==1), anomaly_score )
print(auc_roc)
# %%
#
# Take a look at the MG time series: zoom into the first anomaly
#
#plot_results_swat(test_X, test_labels, anomaly_score, pl_range = (61000, 63000), plot_signal = True, plot_anomaly_score = False)

# %%



