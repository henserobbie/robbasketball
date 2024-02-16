import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# load data
data = pd.read_csv('data/processed/marchGames.csv')

# get x and Y datasets
columns = data.columns
filterSet = set(['Season', 'TeamID1', 'Score1', 'TeamID2', 'Score2'])
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

# Grid seach for best parameters
grid_params = {
    'n_neighbors': [1,11,21,31,41,51,71,101],
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'weights': ['uniform', 'distance'],
}
KNN_GV = GridSearchCV(KNeighborsClassifier(), grid_params, cv = 5)
KNN_GV.fit(X_train, y_train)

# #test
print(f'The best parameters are {KNN_GV.best_params_}')
print(f'The best accuracy on the training data is {KNN_GV.score(X_train, y_train)}')
print(f'The best accuracy on the testing data is {KNN_GV.score(X_test, y_test)}')
