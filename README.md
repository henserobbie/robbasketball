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

## KNN
I applied KNN to the data, but it yields significantly worse test accuracy than the neural network, so it is not used.

## Deep Learning Model
The model for now is a simple neural network with 3 fully connected layers. It takes in the season stats of each team and outputs the win probabilities for each team. In the future, we could skip the data preprocessing and input the box scores of each teams' regular season games to an encoder network and use an attention based deep network to decode the endocer outputs. This would provide the model with more data rather than season averages.

## Usage
- Download the data specified in the Data section.
- Run `python src/preEfficiencies.py` to iteratively compute the adjusted offective, defensive, tempo efficiencies.
- Run `python src/preAvgs.py` to calculate basic season averages.
- Run `python src/compileData.py` to attach each team's season averages to each march madness matchup.
- Run `python src/neuralNetwork.py` to train the network.
- Run `python src/pred.py` to start making predictions. Enter the season and team ids to make a prediction. The output is the probability that team 1 wins.