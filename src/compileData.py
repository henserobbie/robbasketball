import pandas as pd
import os

EFFICIENCYITER = 20

# import data
basePath = os.getcwd()
march = pd.read_csv(f'{basePath}/data/raw/MNCAATourneyCompactResults.csv',dtype = int, converters={'WLoc':str})
teamStats = pd.read_csv(f'{basePath}/data/processed/teamStats.csv', dtype = float, \
                        converters = {'Season':int, 'TeamID':int})

# remove uncessary columns and rows, rename columns
march = march[(march['Season'] >= 2003)]
march = march.drop(['DayNum', 'WLoc', 'NumOT'], axis=1)
march = march.rename(columns={'Season':'Season', 'WTeamID':'TeamID1', 'WScore':'Score1', 'LTeamID':'TeamID2', 'LScore':'Score2'})

# add team stats
stats = ['OE', 'DE', 'Tempo', 'EFGP', 'TOP', 'SBP', 'REBP', 'ASTP', 'FD']
for stat in stats:
    march[stat+'1'] = march.apply(lambda row: \
                    teamStats.loc[(teamStats['Season'] == row['Season']) & (teamStats['TeamID'] == row['TeamID1']), stat].values[0], axis=1)
    march[stat+'2'] = march.apply(lambda row: \
                    teamStats.loc[(teamStats['Season'] == row['Season']) & (teamStats['TeamID'] == row['TeamID2']), stat].values[0], axis=1)

# duplicate and switch teams 1 and 2
switched = march.copy()
switched[[stat+'1' for stat in stats] + [stat+'2' for stat in stats] + ['TeamID1', 'Score1', 'TeamID2', 'Score2']] = \
    march[[stat+'2' for stat in stats] + [stat+'1' for stat in stats] + ['TeamID2', 'Score2', 'TeamID1', 'Score1']]
marchFinal = pd.concat([march, switched], ignore_index=True)

# save march games
marchFinal.to_csv(f'{basePath}/data/processed/marchGames.csv', index=False)