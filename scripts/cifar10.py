import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten the image data
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# Concatenate labels and flattened image data
train_data = np.concatenate((y_train, x_train_flattened), axis=1)
test_data = np.concatenate((y_test, x_test_flattened), axis=1)

# Create DataFrames without headers
df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Save to CSV without headers
df_train.to_csv('cifar10_train.csv', index=False, header=False)
df_test.to_csv('cifar10_test.csv', index=False, header=False)