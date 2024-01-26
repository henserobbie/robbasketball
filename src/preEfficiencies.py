import pandas as pd
import os

EFFICIENCYITER = 20

# import data
basePath = os.getcwd()
games = pd.read_csv(f'{basePath}/data/raw/MRegularSeasonDetailedResults.csv',dtype = float,\
                       converters = {'Season':int, 'DayNum':int, 'WTeamID':int, 'LTeamID':int, 'WLoc':str})
# rankData = pd.read_csv(f'{basePath}/data/raw/MMasseyOrdinals_thru_Season2023_Day128.csv',\
#             converters = {'Season':int, 'RankingDayNum':int, 'SystemName':str, 'TeamID':int, 'OrdinalRank':int})
# rankData = rankData[rankData['SystemName'] == 'POM']
teams = pd.read_csv('data/raw/MTeams.csv',\
                    converters = {'TeamID':int, 'TeamName':str, 'FirstD1Season':int, 'LastD1Season':int})

#skeleton of processed data
teamDict = {row['TeamID']: {'begin': max(row['FirstD1Season'], 2003), 'end': row['LastD1Season']} for index, row in teams.iterrows()}
teamStats = pd.DataFrame(columns = ['Season','TeamID', 'OE', 'DE', 'Tempo'])
for t, info in teamDict.items():
    for s in range(info['begin'], info['end']+1):
        if len(games[(games['Season'] == s) & ((games['WTeamID'] == t) | (games['LTeamID'] == t))].index) > 0:
            teamStats.loc[len(teamStats.index)] = {'Season': s, 'TeamID': t}

# calculate poss per game
games['WPos'] = games['WFGA'] - games['WOR'] + games['WTO'] + 0.475 * games['WFTA']
games['LPos'] = games['LFGA'] - games['LOR'] + games['LTO'] + 0.475 * games['LFTA']

# game raw game efficiencies
games['WOERaw'] = 100 * games['WScore'] / games['WPos']
games['LOERaw'] = 100 * games['LScore'] / games['LPos']
games['WDERaw'] = games['LOERaw']
games['LDERaw'] = games['WOERaw']
games['WTempoRaw'] = games['WPos'] / (40 + 5 * games['NumOT'])
games['LTempoRaw'] = games['LPos'] / (40 + 5 * games['NumOT'])

# calculate base adj efficiencies
for index, row in teamStats.iterrows(): #init
    t = row['TeamID']
    s = row['Season']
    fdf1 = games[(games['Season'] == s) & (games['WTeamID'] == t)]
    fdf2 = games[(games['Season'] == s) & (games['LTeamID'] == t)]
    if len(fdf1.index) > 0 and len(fdf2.index) > 0:
        avgOE = (fdf1['WOERaw'].mean() * len(fdf1.index) + fdf2['LOERaw'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
        avgDE = (fdf1['WDERaw'].mean() * len(fdf1.index) + fdf2['LDERaw'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
        avgTempo = (fdf1['WTempoRaw'].mean() * len(fdf1.index) + fdf2['LTempoRaw'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
    elif len(fdf1.index) == 0:
        avgOE = fdf2['LOERaw'].mean()
        avgDE = fdf2['LDERaw'].mean()
        avgTempo = fdf2['LTempoRaw'].mean()
    else:
        avgOE = fdf1['WOERaw'].mean()
        avgDE = fdf1['WDERaw'].mean()
        avgTempo = fdf1['WTempoRaw'].mean()
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'OE'] = avgOE
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'DE'] = avgDE
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'Tempo'] = avgTempo

# iteratively converge adjusted efficiencies
for s in range(2003, 2024):
    for i in range(EFFICIENCYITER):
        natAvgOE = teamStats[teamStats['Season'] == s]['OE'].mean()
        natAvgDE = teamStats[teamStats['Season'] == s]['DE'].mean()
        natAvgTempo = teamStats[teamStats['Season'] == s]['Tempo'].mean()
        # reduce games to single season and calculate adj efficiencies per game
        fdf = games[games['Season'] == s]
        fdf = fdf.assign(WOEAdj = fdf.apply(lambda row:
            row['WOERaw'] * natAvgDE / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['LTeamID']), 'DE'].values[0],
        axis=1))
        fdf = fdf.assign(LOEAdj = fdf.apply(lambda row:
            row['LOERaw'] * natAvgDE / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['WTeamID']), 'DE'].values[0],
        axis=1))
        fdf = fdf.assign(WDEAdj = fdf.apply(lambda row:
            row['WDERaw'] * natAvgOE / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['LTeamID']), 'OE'].values[0],
        axis=1))
        fdf = fdf.assign(LDEAdj = fdf.apply(lambda row:
            row['LDERaw'] * natAvgOE / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['WTeamID']), 'OE'].values[0],
        axis=1))
        fdf = fdf.assign(WTempoAdj = fdf.apply(lambda row:
            row['WTempoRaw'] * natAvgTempo / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['LTeamID']), 'Tempo'].values[0],
        axis=1))
        fdf = fdf.assign(LTempoAdj = fdf.apply(lambda row:
            row['LTempoRaw'] * natAvgTempo / teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == row['WTeamID']), 'Tempo'].values[0],
        axis=1))
        # calc adj eff avgs per team
        for index, row in teamStats[teamStats['Season'] == s].iterrows():
            t = row['TeamID']
            fdf1 = fdf[(fdf['WTeamID'] == t)]
            fdf2 = fdf[(fdf['LTeamID'] == t)]
            if len(fdf1.index) > 0 and len(fdf2.index) > 0:
                avgOE = (fdf1['WOEAdj'].mean() * len(fdf1.index) + fdf2['LOEAdj'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
                avgDE = (fdf1['WDEAdj'].mean() * len(fdf1.index) + fdf2['LDEAdj'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
                avgTempo = (fdf1['WTempoAdj'].mean() * len(fdf1.index) + fdf2['LTempoAdj'].mean() * len(fdf2.index)) / (len(fdf1.index) + len(fdf2.index))
            elif len(fdf1.index) == 0:
                avgOE = fdf2['LOEAdj'].mean()
                avgDE = fdf2['LDEAdj'].mean()
                avgTempo = fdf2['LTempoAdj'].mean()
            else:
                avgOE = fdf1['WOEAdj'].mean()
                avgDE = fdf1['WDEAdj'].mean()
                avgTempo = fdf1['WTempoAdj'].mean()
            teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'OE'] = avgOE
            teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'DE'] = avgDE
            teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'Tempo'] = avgTempo
        print(f'Season: {s}:{i / EFFICIENCYITER * 100}%')

# save team stats
teamStats.to_csv(f'{basePath}/data/processed/efficiencies.csv', index=False)