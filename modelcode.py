!pip install pandas numpy torch transformers flask plotly leaflet twilio opencv-python requests scikit-learn


from google.colab import files
files.upload()
import pandas as pd


# Create a sample dataset
data = {
    'timestamp': ['2022-01-01 00:00:00', '2022-01-01 00:01:00', '2022-01-01 00:02:00'],
    'latitude': [37.7749, 37.7751, 37.7753],
    'longitude': [-122.4194, -122.4192, -122.4190],
    'mmsi': [123456789, 987654321, 111111111]
}


# Create a DataFrame from the sample dataset
ais_data = pd.DataFrame(data)


# Print the sample dataset
print(ais_data)
ais_data = pd.DataFrame({
    'timestamp': [],
    'latitude': [],
    'longitude': [],
    'mmsi': []
})


from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import zipfile
import pandas as pd
import os

# Extract the zip file
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Get the list of files in the extracted directory
files = os.listdir()

# Print the list of files
print(files)

# Read the data from the files
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        print(df.head())
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
print(files)
import os
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

print(os.getcwd())
dfs = []
for file in files:
    df = pd.read_csv(file)
    print(df.head())
    dfs.append(df)
import zipfile
import pandas as pd
import os

with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
print(files)

dfs = []
for file in files:
    df = pd.read_csv(file)
    print(df.head())
    dfs.append(df)

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
else:
    print("No dataframes to concatenate")
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Extract the zip file
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Get the list of files in the extracted directory
files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

# Initialize an empty list to store the dataframes
dfs = []

# Read the data from the files
for file in files:
    try:
        df = pd.read_csv(file)
        print(df.head())
        dfs.append(df)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

# Concatenate the dataframes
if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
else:
    print("No dataframes to concatenate")



# Perform some basic…
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Extract the zip file
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()


# Get the list of files in the extracted directory
files = [f for f in zip_ref.namelist() if f.endswith('.csv')]


# Initialize an empty list to store the dataframes
dfs = []


# Read the data from the files
for file in files:
    try:
        df = pd.read_csv(file)
        print(df.head())
        dfs.append(df)
    except Exception as e:
        print(f"Error reading file {file}: {e}")


# Check if the list of dataframes is empty
if not dfs:
    print("No dataframes to concatenate")
else:
    # Concatenate the dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
import rasterio
import zipfile

# Extract the zip file
with zipfile.ZipFile('AISVesselTransitCounts2023.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Get the list of files in the extracted directory
files = [f for f in zip_ref.namelist() if f.endswith('.tif')]

# Read the GeoTIFF files
for file in files:
    try:
        with rasterio.open(file) as src:
            print(src.meta)
            print(src.read(1))
    except Exception as e:
        print(f"Error reading file {file}: {e}")


import numpy as np
import pandas as pd

# Create a dummy satellite dataset
np.random.seed(0)
satellite_data = np.random.rand(100, 100, 3)  # 100x100x3 array

# Create a pandas DataFrame to store the satellite data
df = pd.DataFrame({
    'latitude': np.random.uniform(-90, 90, size=100),
    'longitude': np.random.uniform(-180, 180, size=100),
    'satellite_data': [satellite_data[i, :, :] for i in range(100)]
})

# Save the DataFrame to a CSV file
df.to_csv('satellite_data.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dummy satellite dataset
df = pd.read_csv('satellite_data.csv')

# Convert the satellite data column to a numerical array
df['satellite_data'] = df['satellite_data'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', '').replace(',', ' '), dtype=float, sep=' '))

# Visualize the satellite data
plt.imshow(df['satellite_data'][0].reshape(100, 3), cmap='gray')
plt.show()


import pandas as pd

# Read the dataset
data = pd.read_csv('satellite_data.csv')


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X = data
y = np.random.randint(0, 2, size=len(data))  # Create a dummy target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a one-hot encoder
encoder = OneHotEncoder()

# Fit the encoder to the data
encoder.fit(X_train)

# Transform the data using the encoder
X_train_encoded = encoder.transform(X_train)

# Convert the encoded data to a DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray())


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a one-hot encoder
encoder = OneHotEncoder()

# Reshape the X_train variable to a 1D array
X_train_reshaped = X_train.values.flatten()

# Convert all values to strings
X_train_reshaped = X_train_reshaped.astype(str)

# Fit the encoder to the data
encoder.fit(X_train_reshaped.reshape(-1, 1))




import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a label encoder
encoder = LabelEncoder()

# Fit the encoder to the data
encoder.fit(X_train_reshaped)


X_train_transformed = encoder.transform(X_train_reshaped.reshape(-1, 1))



# Assuming X_train is your original feature data before flattening
X_train_transformed = X_train.copy()  # Create a copy to avoid modifying original data

# Apply LabelEncoder to each column (feature) separately
for col in X_train_transformed.columns:
    if X_train_transformed[col].dtype == 'object':  # Encode only categorical features
        le = LabelEncoder()
        X_train_transformed[col] = le.fit_transform(X_train_transformed[col])

y_train = np.array(y_train)

# Check if X_train_transformed and y_train have the same number of samples
if X_train_transformed.shape[0] != len(y_train):
    raise ValueError("X_train_transformed and y_train must have the same number of samples")

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_transformed, y_train, test_size=0.2, random_state=42
)





from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Make predictions on the test set
y_pred = model.predict(X_test_split)

# Evaluate the model
accuracy = accuracy_score(y_test_split, y_pred)
precision = precision_score(y_test_split, y_pred)
recall = recall_score(y_test_split, y_pred)
f1 = f1_score(y_test_split, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


import joblib



import numpy as np

# Apply the string operations to specific columns instead of the entire DataFrame
for column in ['latitude', 'longitude', 'satellite_data']:  # Replace with your actual column names
    X_train[column] = X_train[column].astype(str).apply(lambda x: x.replace('[', '').replace(']', '')).str.split()

X_train = X_train.applymap(lambda x: np.fromstring(' '.join(x) if isinstance(x, list) else x, dtype=float, sep=' ') if isinstance(x, (str, list)) else x)




import numpy as np

# Convert the DataFrame to a NumPy array before processing
X_train_array = X_train.values

# Apply the string operations to specific columns instead of the entire DataFrame
for column_index in [X_train.columns.get_loc(col) for col in ['latitude', 'longitude', 'satellite_data']]:  # Replace with your actual column names
    X_train_array[:, column_index] = [np.fromstring(str(val).replace('[', '').replace(']', ''), dtype=float, sep=' ') for val in X_train_array[:, column_index]]

# Update X_train with the modified array
X_train = pd.DataFrame(X_train_array, columns=X_train.columns)




import numpy as np

# Your numpy array
X_train = np.array([np.fromstring(x, dtype=float, sep=' ') for x in X_train])

# Save the numpy array to a file
np.save('X_train.npy', X_train)

# Alternatively, you can use np.savez to save the numpy array to a compressed file
np.savez_compressed('X_train.npz', X_train)





import pickle as pkl

# Save the numpy array to a file using pickle
with open('X_train.pkl', 'wb') as f:
    pkl.dump(X_train, f)

# Load the numpy array from the file
X_train_loaded = np.load('X_train.npy')

# Alternatively, you can use np.load to load the compressed file
X_train_loaded = np.load('X_train.npz')['arr_0']

# Or, you can use pickle to load the file
with open('X_train.pkl', 'rb') as f:
    X_train_loaded = pkl.load(f)

import os
print(os.getcwd())

np.save('my_file.npy', X_train)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Create a model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model in h5 format
model.save('my_model.h5')

model.save_weights('my_model_weights.weights.h5')

from keras.models import load_model

loaded_model = load_model('my_model.h5')

from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np

# Create a model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model in h5 format
model.save('my_model.h5')

# Load the model from the file
loaded_model = load_model('my_model.h5')

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import os
print(os.getcwd())

model.save('/path/to/your/model.h5')

# Save the model in a specific location
model.save('/Users/username/Documents/my_model.h5')

'/content/my_model.h5'
