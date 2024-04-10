import pandas as pd
import os

EFFICIENCYITER = 20

# import data
basePath = os.getcwd()
games = pd.read_csv(f'{basePath}/data/raw/MRegularSeasonDetailedResults.csv',dtype = float,\
                       converters = {'Season':int, 'DayNum':int, 'WTeamID':int, 'LTeamID':int, 'WLoc':str})
teamStats = pd.read_csv(f'{basePath}/data/processed/efficiencies.csv', \
                        converters = {'Season':int, 'TeamID':int, 'OE':float, 'DE':float, 'Tempo':float})

#add columns for new stats
teamStats['EFGP'] = 0
teamStats['TOP'] = 0
teamStats['SBP'] = 0
teamStats['REBP'] = 0
teamStats['ASTP'] = 0
teamStats['FD'] = 0

# calculate poss per game
games['WPos'] = games['WFGA'] - games['WOR'] + games['WTO'] + 0.475 * games['WFTA']
games['LPos'] = games['LFGA'] - games['LOR'] + games['LTO'] + 0.475 * games['LFTA']

n = len(teamStats.index)
prevPercent = 0

for index, row in teamStats.iterrows():
    t = row['TeamID']
    s = row['Season']
    fdf1 = games[(games['Season'] == s) & (games['WTeamID'] == t)]
    fdf2 = games[(games['Season'] == s) & (games['LTeamID'] == t)]
    n1 = len(fdf1.index)
    n2 = len(fdf2.index)
    #effective field goal percentage
    efgp = ((fdf1['WFGM'].sum() + fdf2['LFGM'].sum()) + 0.5*(fdf1['WFGM3'].sum() + fdf2['LFGM3'].sum())) / (fdf1['WFGA'].sum() + fdf2['LFGA'].sum())
    #turnover percentage
    top = (fdf1['WTO'].sum() + fdf2['LTO'].sum()) / (fdf1['WPos'].sum() + fdf2['LPos'].sum())
    #steal/block percentage
    sbp = (fdf1['WStl'].sum() + fdf2['LStl'].sum() + fdf1['WBlk'].sum() + fdf2['LBlk'].sum()) / (fdf1['LPos'].sum() + fdf2['WPos'].sum())
    #rebound percentage
    myReb = fdf1['WOR'].sum() + fdf2['LOR'].sum() + fdf1['WDR'].sum() + fdf2['LDR'].sum()
    oppReb = fdf1['LOR'].sum() + fdf2['WOR'].sum() + fdf1['LDR'].sum() + fdf2['WDR'].sum()
    rebp = myReb / (myReb + oppReb)
    #assist percentage
    astp = (fdf1['WAst'].sum() + fdf2['LAst'].sum()) / (fdf1['WFGM'].sum() + fdf2['LFGM'].sum())
    #avg foul differential
    fd = (fdf1['WPF'].sum() - fdf1['LPF'].sum() + fdf2['LPF'].sum() - fdf2['WPF'].sum()) / (n1 + n2)
    #apply stats
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'EFGP'] = efgp
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'TOP'] = top
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'SBP'] = sbp
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'REBP'] = rebp
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'ASTP'] = astp
    teamStats.loc[(teamStats['Season'] == s) & (teamStats['TeamID'] == t), 'FD'] = fd
    #status output
    newPercent = index / n * 100
    if newPercent >= prevPercent + 5:
        prevPercent += 5
        print(f'Status: {prevPercent}%')

#save team stats
teamStats.to_csv(f'{basePath}/data/processed/teamStats.csv', index=False)