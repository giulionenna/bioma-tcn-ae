# %%
import numpy
import time
from utilities import select_gpus, plot_results # utilities.py: Contains a few miscellaneous functions 
from tcnae import TCNAE # tcnae.py: Specification of the TCN-AE model
import data_swat # data.py: Allows to generate anomalous Mackey-Glass (MG) time series 

# If you have several GPUs, select one or more here (in a list)
#select_gpus(0)

# %%
train_ts_id = 1 # [1-10]. Train the model on Mackey-Glass time series 1
data_gen = data_swat.DataSwat(window_length = 1050, ratio = 0.2)
train_data = data_gen.build_data() # Returns a dictionary
train_X = train_data["train_X"] # We only need train_X (input = output) for the training process
print("train_X.shape:", train_X.shape) # A lot of training sequences of length 1050 and dimension 1

# %%
import keras.backend as K
K.clear_session()

# %%

# Build and compile the model
#
tcn_ae = TCNAE(ts_dimension=train_X.shape[2], dilations=(1, 2, 4, 8, 16)) # Use the parameters specified in the paper

#
# Train TCN-AE for 10 epochs. For a better accuracy 
# on the test case, increase the epochs to epochs=40 
# The training takes about 3-4 minutes for 10 epochs, 
# and 15 minutes for 40 epochs (on Google CoLab, with GPU enabled)
#
tcn_ae.fit(train_X, train_X, batch_size=16, epochs=10, verbose=1)

# %%


# %%
#
# Test the model on another Mackey-Glass time series
# Might take a few minutes...
#
start_time = time.time()
test_ts_id = 3 # Test the model on Mackey-Glass time series 3
test_data = data_gen.build_data(test_ts_id, verbose = 2) # Returns a dictionary

#
# Take the whole time series... Like the training data, the test data is standardized (zero mean and unit variance)
#
test_X = test_data["scaled_series"].values[numpy.newaxis,:,:] # We need an extra dimension for the batch-dimension
print("test_X.shape", test_X.shape) # This is one long time series
anomaly_score = tcn_ae.predict(test_X)
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
plot_results(test_data, anomaly_score, pl_range = None, plot_signal = False, plot_anomaly_score = True)

# %%
#
# Take a look at the MG time series: zoom into the first anomaly
#
plot_results(test_data, anomaly_score, pl_range = (61000, 63000), plot_signal = True, plot_anomaly_score = False)

# %%



