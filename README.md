# robbasketball
March Madness bracket predictor using python data science. Using multiple models for Game Winner, Spread, Total. May have sports betting applications if backtests yield profits. Work in progress.

## Data
Raw data must be downloaded from https://www.kaggle.com/competitions/march-machine-learning-mania-2023 and placed in:
```
data/raw/
```
The required datasets from kaggle are:
```
MNCAATourneyCompactResults.csv
MNCAATourneySeeds.csv
MRegularSeasonDetailedResults.csv
MTeams.csv
```

## Preprocessing
We want to compile regular season box scores into a stat list for each team each season. These stats are used to train a model to predict march madness game outcomes based on regular season stats of each team. We have 9 stats to calculate from regular season data:
* **Adjusted Offensive Efficiency**: Average points scored over 100 possessions, adjusted for opponent adj defensive efficiency.
* **Adjusted Defensive Efficiency**: Average points allowed over 100 opponent possessions, adjusted for opponent adj offensive efficiency.
* **Adjusted Tempo**: Average possessions per 40 minutes, adjusted for opponent adj tempo.
* **Effective Field Goal %**: Field goal percentage with a 50% higher weighting for 3-pointers.
* **Turnover %**: Percentage of possessions that result in a turnover.
* **Steal + Block %**: Percentage of opponent possessions that result in a steal or block.
* **Rebound %**: Avg percentage of rebound secured by the team.
* **Assist %**: Percent of field goals that credit an assist.
* **Foul Differential**: Average of (my fouls) - (opponent fouls) over the season.

## PCA
When we train the model for march madness games, we would have 18 dimensions of input data (9 stats for each team). I conducted a principle component analysis to see if I could reduce the dimensionality of the data. This was the result:
![](/docs/pca.png)
As shown in the graph, there is not a clear elbow that indicates a point where you can significantly reduce the dimensionality while retaining 90-95% of the variance of the data. The next step is to explore whether it is plausible to run a KNN algorithm in 18 dimensions. I graphed the distribution of the distances between points in the training set. This was the result:
![](/docs/dimensionality.png)
As seen above, the distribution has enough variance to avoid the curse of dimensionality, meaning a KNN classification model can be used on the data.

## Tuning Model Parameters
To be completed

## Final Model(s)
To be completed

## Generate Bracket(s)
To be completed