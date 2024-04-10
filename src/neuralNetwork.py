import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load data
data = pd.read_csv('data/processed/marchGames.csv')

# Get x and y datasets
columns = data.columns
filterSet = set(['TeamID1', 'Score1', 'TeamID2', 'Score2'])
xColumns = [n for n in columns if n not in filterSet]
X = data[xColumns]
y = data['Score1'] > data['Score2']

# Swap order to remove bias
switched = X.copy()
stats = ['OE', 'DE', 'Tempo', 'EFGP', 'TOP', 'SBP', 'REBP', 'ASTP', 'FD']
switched[[stat+'1' for stat in stats] + [stat+'2' for stat in stats]] = \
    X[[stat+'2' for stat in stats] + [stat+'1' for stat in stats]]
X = pd.concat([X, switched], ignore_index=True)

# Create labels for both teams winning
y_inverse = ~y

# Concatenate original and modified labels
y = pd.concat([y, y_inverse], ignore_index=True)

# Standardize data
sc = StandardScaler() 
X = sc.fit_transform(X)

# Create network
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Change output layer to predict two classes
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train network
model.fit(X, y, epochs=10, batch_size=32)

# Save model
model.save('model.h5')
scaler_filename = "scaler.save"
joblib.dump(sc, scaler_filename)
