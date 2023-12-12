import torch
import time
import cfbd
from cfbd.rest import ApiException
from pprint import pprint
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# path to save model to
PATH = f'models/L1_adam_2016_2023.tar'

# import matchup histories, inputs and outputs
gameInputsMat = loadmat('gameInputs_2016_2023_clean.mat')
gameOutputsMat = loadmat('gameOutputs_2016_2023_full.mat')

# import stats for seasons 2016-2023
stats = loadmat('stats_2016_2023_clean.mat')
stats = np.array(stats['stats_2016_2023'])

# import list of teams
fbsTeams = loadmat('teams.mat')
fbsTeams = [team.strip() for team in fbsTeams['teams']]

X_raw = np.array(gameInputsMat['gameInputs'])
y = np.array(gameOutputsMat['gameOutputs'])

# let y represent the difference in home points minus away points for each game
y = np.subtract(y[:, 0], y[:, 1])
# y = np.dot(y, [[1, 1], [-1, 1]])

numSeasons = 8
initialYear = 2016
numTeams = len(fbsTeams)
numGames, numStats = np.shape(X_raw)
numStats -= 1
numStatsPerTeam = int(numStats/2)

# y = np.zeros((numGames, 2))

# # [home pts., away pts.]
# for sc in range(len(y_)):
#     y[sc//2][sc % 2] = y_[sc]

# # [pt. diff., over/under]
# y[:, 0] = y_[:, 0] - y_[:, 1]
# y[:, 1] = y_[:, 0] + y_[:, 1]

X_norm = np.zeros((numGames, numStats))
X_temp = np.zeros((numGames, numStats))

stats_norm = np.zeros((numTeams, numStatsPerTeam, numSeasons))

# normalize (z-score) stats by season
for n in range(numSeasons):
    if n == 4: continue
    seasonStats = stats[:, :, n]
    for s in range(numStatsPerTeam):
        if s == 11 or s == 38:
            seasonStats[:, s] /= seasonStats[:, s + 1]  # normalize 3rd/4th down conv.

        meanS = np.mean(seasonStats[:, s])
        stdS = np.std(seasonStats[:, s])
        if stdS == 0: stdS = 1
        for t in range(numTeams):
            if all(seasonStats[t, :] == np.zeros_like(seasonStats[t, :])): 
                stats_norm[t, :, n] = np.zeros_like(stats_norm[t, :, n])
                continue
            # normalize dataset of stats by season by team
            stats_norm[t, s, n] = (seasonStats[t, s] - meanS)/stdS

    s_ = 0
    seasonGames = [i for i in range(numGames) if X_raw[i, 0] == n + initialYear]
    for s in range(numStats):
        if s == 12 or s == 39 or s == 168 or s == 194:
            X_raw[seasonGames, s] /= X_raw[seasonGames, s + 1]  # normalize 3rd/4th down conv.

        meanS = np.mean(X_raw[seasonGames, s])
        stdS = np.std(X_raw[seasonGames, s])
        if stdS == 0: stdS = 1
        for g in seasonGames:
            # ensure matchup stats are normalized
            X_norm[g, s_] = (X_raw[g, s] - meanS)/stdS
            X_temp[g, s_] = X_raw[g, s]
            # if s == 0: X_norm[g, s] = X_raw[g, s]
        
        s_ += 1

stats_norm = np.nan_to_num(stats_norm)

# possible combinations of stats for consideration

# miscStats = [0, 138, 3, 4, 5, 9, 11, 12, 14, 18, 20, 28, 29, 36, 37, 38, 39, 41, 143, 151, 152] # fpi, talent, rank SOS, rank game control, overall eff., elo, 4th conv., num. 4th, fumbles rec., ints., KR yds., penalty yds., poss. time, sacks, tfl, 3rd conv., num. 3rd, turnovers, returning ppa percent, coach srs, coach sp overall
# offStats = [44, 46, 47, 49, 50, 54, 57, 58, 61, 62] # ppa, succ. rate, explosiveness, stuff rate, line yds., open field yds., ppo, start field pos., havoc front 7, havoc out, 
# defStats = [x + 39 for x in offStats]

# miscStats = [0, 138, 3, 5, 11, 37, 41, 38] # fpi, talent, rank SOS, overall eff., 4th conv., tfl, turnovers, 3rd conv.
# offStats = [44, 47, 49, 50, 54, 57, 58, 61, 62] # ppa, explosiveness, stuff rate, line yds., open field yds., ppo, start field pos., havoc front 7, havoc out, 
# defStats = [x + 39 for x in offStats]

# miscStats = list(range(6)) + list(range(8, 15)) + list(range(16, 42)) + list(range(138, 153))
# offStats = [6] + list(range(42, 81)) + list(range(120, 128)) + [153]
# defStats =  [7] + list(range(81, 120)) + list(range(129, 138)) + [154]

miscStats = [0, 3, 5, 8, 9, 138, 151, 152]
offStats = [6, 44, 153]
defStats = [7, 83, 154]

highlightedFeaturesH = miscStats + offStats + defStats
highlightedFeaturesA = miscStats + defStats + offStats

# ignore above selection, consider all stats, do not align offensive and defensive units for comparison
highlightedFeaturesH = list(range(numStatsPerTeam))
highlightedFeaturesA = list(range(numStatsPerTeam))
# selection = highlightedFeatures + [x+numStatsPerTeam for x in highlightedFeatures]


# engineer features by taking difference in home team and away team performance in each stat for each game

# X = X_norm[:, selection]
# X = X_temp[:, 0:numStatsPerTeam-2] - X_temp[:, numStatsPerTeam-2:numStats]
X = np.subtract(X_norm[:, highlightedFeaturesH], X_norm[:, [x+numStatsPerTeam for x in highlightedFeaturesA]])
# X = np.concatenate((X_norm[:, highlightedFeaturesH], X_norm[:, [x+numStatsPerTeam for x in highlightedFeaturesA]]), axis=1)

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = torch.nn.Linear(len(highlightedFeaturesH), 300)
        # self.linear1 = torch.nn.Linear(5, 50)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(.05)

        self.linear2 = torch.nn.Linear(300, 300)
        self.activation2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(300, 25)
        self.activation3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(25, 25)
        self.activation4 = torch.nn.ReLU()

        self.linear5 = torch.nn.Linear(25, 1)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.drop(x)
        x = self.linear1(x)
        x = self.act(x)

        ######### 50 x 50 ##########
        x = self.drop(x)
        x = self.linear2(x)
        x = self.act(x)
        
        x = self.drop(x)
        x = self.linear2(x)
        x = self.act(x)
        
        x = self.drop(x)
        x = self.linear2(x)
        x = self.act(x)
        ############################

        x = self.drop(x)
        x = self.linear3(x)
        x = self.act(x)

        ######### 20 x 20 ##########
        x = self.drop(x)
        x = self.linear4(x)
        x = self.act(x)

        x = self.drop(x)
        x = self.linear4(x)
        x = self.act(x)

        x = self.drop(x)
        x = self.linear4(x)
        x = self.act(x)
        ############################

        x = self.drop(x)
        x = self.linear5(x)
        # x = self.sigmoid1(x)

        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
    # X, y = dataloader.dataset.tensors
        # Compute prediction and loss
        pred = model(X).view(-1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
        # X, y = dataloader.dataset.tensors
            pred = model(X).view(-1)
            # test_loss, correct = 0, 0
            test_loss += loss_fn(pred, y).item()
            pred_sign = torch.sign(pred)
            y_sign = torch.sign(y)
            correct += (pred_sign == y_sign).type(torch.float).sum().item()
            # correct += (pred==y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

model = Net()
# model.load_state_dict(torch.load(PATH))

learning_rate = .00003
batch_size = 32
epochs = 150
testFrac = .2

lossF = torch.nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

# y = np.maximum(0, y/abs(y))
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac, shuffle=True)

trainSet = TensorDataset(X_train, y_train)
testSet = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
# train_dataloader = DataLoader(trainSet, shuffle=True)
# test_dataloader = DataLoader(testSet, shuffle=True)

minModelLoss = test_loop(test_dataloader, model, lossF)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, lossF, opt)
    modelLoss = test_loop(test_dataloader, model, lossF)
    if modelLoss < minModelLoss:
        torch.save(model.state_dict(), PATH)
        minModelLoss = modelLoss

#######################################
#######################################
#######################################
#######################################

##### RANKING THE 2019 CFB SEASON #####
initialYear = 2016
year = 2023 - initialYear

ap25 = ['Georgia', 'Michigan', 'Ohio State', 'Florida State', 'Washington', 'Oregon', 'Texas', 'Alabama', 'Penn State', 'Ole Miss', 'Louisville', 'Oregon State', 'Utah', 'Tennessee', 'Oklahoma State', 'Missouri', 'Oklahoma', 'LSU', 'Kansas', 'Tulane', 'Notre Dame', 'Arizona', 'North Carolina']

# sets of games to print predictions for
week9_home = ['Wake Forest', 'Kansas', 'Penn State', 'Florida', 'Texas', 'Utah', 'Notre Dame', 'Louisville', 'Rice', 'California', 'Stanford', 'Colorado State', 'Kentucky', 'Wisconsin', 'Ole Miss', 'UCLA', 'Georgia Tech', 'Arizona']
week9_away = ['Florida State', 'Oklahoma', 'Indiana', 'Georgia', 'BYU', 'Oregon', 'Pittsburgh', 'Duke', 'Tulane', 'USC', 'Washington', 'Air Force', 'Tennessee', 'Ohio State', 'Vanderbilt', 'Colorado', 'North Carolina', 'Oregon State']

week10_home = ['Georgia', 'Alabama', 'Tennessee', 'Ole Miss', 'Florida', 'Mississippi State', 'Vanderbilt', 'Michigan', 'Rutgers', 'Maryland', 'Indiana', 'Northwestern', 'Michigan State', 'Minnesota', 'USC', 'Utah', 'Oregon', 'Colorado', 'Arizona', 'Washington State', 'Texas Tech', 'Texas', 'Oklahoma State', 'Baylor', 'Cincinnati', 'Iowa State', 'West Virginia', 'Clemson', 'Duke', 'Syracuse', 'Pittsburgh', 'Virginia', 'Louisville', 'Toledo', 'Akron', 'Bowling Green']
week10_away = ['Missouri', 'LSU', 'Connecticut', 'Texas A&M', 'Arkansas', 'Kentucky', 'Auburn', 'Purdue', 'Ohio State', 'Penn State', 'Wisconsin', 'Iowa', 'Nebraska', 'Illinois', 'Washington', 'Arizona State', 'California', 'Oregon State', 'UCLA', 'Stanford', 'TCU', 'Kansas State', 'Oklahoma', 'Houston', 'UCF', 'Kansas', 'BYU', 'Notre Dame', 'Wake Forest', 'Boston College', 'Florida State', 'Georgia Tech', 'Virginia Tech', 'Buffalo', 'Kent State', 'Ball State']

bowl_home = ['Ohio', 'Florida A&M', 'Louisiana', 'Appalachian State', 'Fresno State', 'Boise State', 'Texas Tech', 'Old Dominion', 'Marshall', 'Syracuse', 'UCF', 'Duke', 'Northern Illinois', 'Air Force', 'Utah State', 'Eastern Michigan', 'Northwestern', 'San Jose State', 'Minnesota', 'Rice', 'UNLV', 'Tulane', 'West Virginia', 'USC', 'Oklahoma State', 'Boston College', 'Miami', 'Kansas State', 'Oklahoma', 'Kentucky', 'Notre Dame', 'Iowa State', 'Maryland', 'Wyoming', 'LSU', 'Tennessee', 'Ohio State', 'Penn State', 'Florida State', 'Oregon', 'Michigan', 'Washington']
bowl_away = ['Georgia Southern', 'Howard', 'Jacksonville State', 'Miami (OH)', 'New Mexico State', 'UCLA', 'California', 'Western Kentucky', 'UT San Antonio', 'South Florida', 'Georgia Tech', 'Troy', 'Arkansas State', 'James Madison', 'Georgia State', 'South Alabama', 'Utah', 'Coastal Carolina', 'Bowling Green', 'Texas State', 'Kansas', 'Virginia Tech', 'North Carolina', 'Louisville', 'Texas A&M', 'SMU', 'Rutgers', 'NC State', 'Arizona', 'Clemson', 'Oregon State', 'Memphis', 'Auburn', 'Toledo', 'Wisconsin', 'Iowa', 'Missouri', 'Ole Miss', 'Georgia', 'Liberty', 'Alabama', 'Texas']

model.load_state_dict(torch.load(PATH))
model.eval()

minModelLoss = test_loop(test_dataloader, model, lossF)

netPts = np.zeros((numTeams, 1))
matches = 0

for t1 in range(numTeams):
    for t2 in range(numTeams):
        if t1 == t2: continue
        # if fbsTeams[t2] not in ap25: continue
        # t1 = list(fbsTeams).index(week10_home[h])
        # t2 = list(fbsTeams).index(week10_away[h])
        
        homeStats = [stats_norm[t1, x, year] for x in highlightedFeaturesH]
        awayStats = [stats_norm[t2, x, year] for x in highlightedFeaturesA]
        # neutralGame = np.subtract(homeStats, 0*np.ones_like(awayStats))
        fullGame = np.subtract(homeStats, awayStats)
        # fullGame = np.concatenate((homeStats, awayStats))

        with torch.no_grad():
            # prediction = torch.round(model(torch.tensor(fullGame, dtype=torch.float))).item()
            prediction = model(torch.tensor(fullGame, dtype=torch.float)).item()
            # neutralPred = model(torch.tensor(neutralGame, dtype=torch.float)).item()
            # prediction = model(torch.tensor(homeStats, dtype=torch.float)).item()
            if fbsTeams[t1] in bowl_home and fbsTeams[t2] == bowl_away[bowl_home.index(fbsTeams[t1])]:
            # if fbsTeams[t1] in ap25 and fbsTeams[t2] in ap25:
                print(fbsTeams[t1] + " v. " + fbsTeams[t2] + ": " + str(prediction))
                # print(fbsTeams[t2] + " v. " + fbsTeams[t1] + ": " + str(predictionA))
                # print(fullGame)
            # if fbsTeams[t1] == "Louisiana":
            # print(fbsTeams[t1] + " v. " + fbsTeams[t2] + ": " + str(neutralPred))
                # print(fullGame)
            if fbsTeams[t2] in bowl_home and fbsTeams[t1] == bowl_away[bowl_home.index(fbsTeams[t2])]:
                print(fbsTeams[t1] + " v. " + fbsTeams[t2] + ": " + str(prediction))
                # print(fullGame)

            # netPts[t1] += prediction/abs(prediction)
            # netPts[t2] -= prediction/abs(prediction)
            netPts[t1] += prediction
            netPts[t2] -= prediction

ascendingRankings = [fbsTeams[int(x)] for x in np.argsort(netPts.flatten())]
rankings = ascendingRankings[::-1]

for team in range(len(rankings)):
    print(str(team+1) + ". " + rankings[team] + "\t\t" + str(netPts[list(fbsTeams).index(rankings[team])]/(2*numTeams - 2)))