import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
data = pd.read_csv('data/processed/marchGames.csv')

# get x and Y datasets
columns = data.columns
filterSet = set(['TeamID1', 'Score1', 'TeamID2', 'Score2'])
xColumns = [n for n in columns if n not in filterSet]
X = data[xColumns]
y = data['Score1'] > data['Score2']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# duplicate and switch teams 1 and 2 for training data to balance out model
switched = X_train.copy()
stats = ['OE', 'DE', 'Tempo', 'EFGP', 'TOP', 'SBP', 'REBP', 'ASTP', 'FD']
switched[[stat+'1' for stat in stats] + [stat+'2' for stat in stats]] = \
    X_train[[stat+'2' for stat in stats] + [stat+'1' for stat in stats]]
X_train = pd.concat([X_train, switched], ignore_index=True)
switched = y_train == False
y_train = pd.concat([y_train, switched], ignore_index=True)

# standardize data
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# create network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train network
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# #test
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
