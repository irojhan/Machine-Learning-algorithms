# import modules
import tensorflow as tf
from tensorflow.keras import *

import gzip, pickle, tensorflow_datasets as TFDatasets

"""
-----------------------------------------------------------------------------------------------
Step 1. Restore the model you saved in Problem 1 [5 points]
-----------------------------------------------------------------------------------------------
"""
model: Model = 

"""
-----------------------------------------------------------------------------------------------
Step 2. Test the restored model with both MNIST and USPS testing samples [10 points]
Set batch size to 256
MNIST accuracy =
USPS accuracy before fine-tune:
-----------------------------------------------------------------------------------------------
"""
with gzip.open('usps.pkl') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    usps_train_set, usps_test_set = u.load()

    x_train, y_train = usps_train_set
    x_train: tf.Tensor = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test, y_test = usps_test_set
    x_test: tf.Tensor = tf.reshape(x_test, (-1, 28, 28, 1))
    
mnist = 

model.evaluate()

model.evaluate()

"""
-----------------------------------------------------------------------------------------------
Step 3. Train you CNN with USPS training samples [10 points]
Set epochs to 5 and batch size to 256
-----------------------------------------------------------------------------------------------
"""
# fine-tune
model.fit()

"""
-----------------------------------------------------------------------------------------------
Step 4. Test your fine tuned CNN on USPS testing data and report testing accuracy [5 points]
USPS accuracy after fine-tune =
-----------------------------------------------------------------------------------------------
"""
model.evaluate()
