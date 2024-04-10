import pandas as pd
import pickle
import os
import joblib
from tensorflow.keras.models import load_model

#load data
basePath = os.getcwd()
teamStats = pd.read_csv(f'{basePath}/data/processed/teamStats.csv', dtype = float, \
                        converters = {'Season':int, 'TeamID':int})
teams = pd.read_csv('data/raw/MTeams.csv',\
                    converters = {'TeamID':int, 'TeamName':str, 'FirstD1Season':int, 'LastD1Season':int})

# Load model
model = load_model('model.h5')
scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)

#to format data for model evel
def getInputRow(s, t1, t2):
    #Season,TeamID1,Score1,TeamID2,Score2,OE1,OE2,DE1,DE2,Tempo1,Tempo2,EFGP1,EFGP2,TOP1,TOP2,SBP1,SBP2,REBP1,REBP2,ASTP1,ASTP2,FD1,FD2
    ts1 = teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t1)].reset_index()
    ts2 = teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t2)].reset_index()
    row = pd.DataFrame()
    row['Season'] = s
    row['OE1'] = ts1['OE']
    row['Season'] = s
    row['OE2'] = ts2['OE']
    row['DE1'] = ts1['DE']
    row['DE2'] = ts2['DE']
    row['Tempo1'] = ts1['Tempo']
    row['Tempo2'] = ts2['Tempo']
    row['EFGP1'] = ts1['EFGP']
    row['EFGP2'] = ts2['EFGP']
    row['TOP1'] = ts1['TOP']
    row['TOP2'] = ts2['TOP']
    row['SBP1'] = ts1['SBP']
    row['SBP2'] = ts2['SBP']
    row['REBP1'] = ts1['REBP']
    row['REBP2'] = ts2['REBP']
    row['ASTP1'] = ts1['ASTP']
    row['ASTP2'] = ts2['ASTP']
    row['FD1'] = ts1['FD']
    row['FD2'] = ts2['FD']
    return row

#main loop
s = int(input('Season: '))
while True:
    try:
        # t1 = int(input('Team 1 ID: '))
        # t2 = int(input('Team 2 ID: '))
        # row = getInputRow(s,t1,t2)
        # X = scaler.transform(row)
        # y = model(X)
        # print(y)
        t1 = input('Team 1 ID: ')
        if t1 == 'e':
            break
        t2 = input('Team 2 ID: ')
        if t2 == 'e':
            break
        t1 = int(t1)
        t2 = int(t2)
        row1 = getInputRow(s, t1, t2)
        row2 = getInputRow(s, t2, t1)  # Swapping the order of teams
        X1 = scaler.transform(row1)
        X2 = scaler.transform(row2)
        prob1 = model.predict(X1)[0][1]  # Probability of team 1 winning
        prob2 = model.predict(X2)[0][0]  # Probability of team 1 winning
        print(f'Probability of Team 1 winning: {prob1}')
        print(f'Probability of Team 1 winning (order swapped): {prob2}')
        print(f'Avg Team1 Win Prob: {(prob1+prob2)/2}')
    except:
        print('FAILED')