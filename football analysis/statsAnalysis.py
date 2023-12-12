from __future__ import print_function
import time
import cfbd
from cfbd.rest import ApiException
from pprint import pprint
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

# Configure API key authorization: ApiKeyAuth
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '8lnl0wSJLCGdvNEsnUKUaKY1On0ADk42w9dCkHwZH3E4OKCuipIdh2jeIQVmLCT1'
configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_stats = cfbd.StatsApi(cfbd.ApiClient(configuration))
api_teams = cfbd.TeamsApi(cfbd.ApiClient(configuration))
api_metrics = cfbd.MetricsApi(cfbd.ApiClient(configuration))
api_games = cfbd.GamesApi(cfbd.ApiClient(configuration))
api_players = cfbd.PlayersApi(cfbd.ApiClient(configuration))
api_ratings = cfbd.RatingsApi(cfbd.ApiClient(configuration))
api_coaches = cfbd.CoachesApi(cfbd.ApiClient(configuration))

final_year = 2023 # int | Year/season filter for games (optional)]
initial_year = final_year - 7

egt = True

statLine = 'stats_'+str(initial_year)+'_'+str(final_year)

def teams(year):
    try:
        return api_teams.get_fbs_teams(year=year)
    except (ApiException, IndexError) as e:
        print("Exception when calling TeamsApi->get_lines: %s\n" % e)
    return

def games(year, team, week):
    try:
        statsNestedL = api_games.get_games(year=year, team=team, week=week)
        if len(statsNestedL) == 0: return {'-1': '-1'}

        statsNested = statsNestedL[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling MetricsApi->get_lines: %s\n" % e)
    return
        
def basicStats(year, team):
    try:
        # print(api_stats.get_advanced_team_season_stats(year=year, team=team))
        statsDicts = api_stats.get_team_season_stats(year=year, team=team)
        k = []
        v = []
        for s in statsDicts:
            s = s.to_dict()
            k += [s['stat_name']]
            v += [s['stat_value']]
        stats = list(zip(k, v))
        stats = sorted(stats, key=lambda x: x[0])
        stats = dict(stats)
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling StatsApi->get_lines: %s\n" % e)
        statsDicts = api_stats.get_team_season_stats(year=year, team=fbsTeams[0])
        k = []
        v = []
        for s in statsDicts:
            s = s.to_dict()
            k += [s['stat_name']]
            v += [0]
        stats = list(zip(k, v))
        stats = sorted(stats, key=lambda x: x[0])
        stats = dict(stats)
        return stats
    
def advStats(year, team):
    try:
        # print(api_stats.get_advanced_team_season_stats(year=year, team=team))
        statsNested = api_stats.get_advanced_team_season_stats(year=year, team=team, exclude_garbage_time=egt)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling StatsApi->get_lines: %s\n" % e)
        statsNested = api_stats.get_advanced_team_season_stats(year=initial_year, team=fbsTeams[0], exclude_garbage_time=egt)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD

def metrics(year, team):
    try:
        statsNested = api_metrics.get_team_ppa(year=year, team=team, exclude_garbage_time=egt)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling MetricsApi->get_lines: %s\n" % e)
        statsNested = api_metrics.get_team_ppa(year=initial_year, team=fbsTeams[0], exclude_garbage_time=egt)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD

def record(year, team):
    try:
        statsNested = api_games.get_team_records(year=year, team=team)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling MetricsApi->get_lines: %s\n" % e)
    return

def findTalent(allTeamDicts, targTeam):
    for teamDict in allTeamDicts:
        teamDict = teamDict.to_dict()
        if teamDict['school'] == targTeam: return [teamDict['talent']]
    return [0]

def talent(year, team):
    try:
        stats = api_teams.get_talent(year=year)
        return findTalent(stats, team)
    except (ApiException, IndexError) as e:
        print("Exception when calling MetricsApi->get_lines: %s\n" % e)
    return [0]

def returningProd(year, team):
    try:
        statsNested = api_players.get_returning_production(year=year, team=team)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling MetricsApi->get_lines: %s\n" % e)
        statsNested = api_players.get_returning_production(year=initial_year, team=fbsTeams[0])[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD

def fpi(year, team):
    try:
        statsNested = api_ratings.get_fpi_ratings(year=year, team=team)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]

        stats = {key: value for key, value in stats.items() if value is not None}
        print(stats)
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling RatingsApi->get_lines: %s\n" % e)
        statsNested = api_ratings.get_fpi_ratings(year=initial_year, team=fbsTeams[0])[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD
    
def elo(year, team):
    try:
        statsNested = api_ratings.get_elo_ratings(year=year, team=team)[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling RatingsApi->get_lines: %s\n" % e)
        statsNested = api_ratings.get_elo_ratings(year=year, team=fbsTeams[0])[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD
    
def coach(year, team):
    try:
        statsNested = api_coaches.get_coaches(year=year, team=team)[0].to_dict()
        stats_df = pd.json_normalize(statsNested['seasons'][0], sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        return stats
    except (ApiException, IndexError) as e:
        print("Exception when calling RatingsApi->get_lines: %s\n" % e)
        statsNested = api_ratings.get_fpi_ratings(year=initial_year, team=fbsTeams[0])[0].to_dict()
        stats_df = pd.json_normalize(statsNested, sep='_')
        stats = stats_df.to_dict(orient='records')[0]
        statsD = dict.fromkeys(stats, 0)
        
        print(statsD)
        return statsD

fbsTeams_full = [team.school for team in teams(final_year)]
newTeams = ['Coastal Carolina', 'Liberty', 'James Madison', 'Sam Houston', 'Jacksonville State', 'UAB', 'Charlotte', 'Appalachian State', 'Georgia Southern', 'Old Dominion', 'Georgia State', 'South Alabama', 'Texas State', 'UMass', 'UT San Antonio', 'Sam Houston State']
fbsTeams = [team for team in fbsTeams_full if team not in newTeams]

print(fbsTeams)

baiscStatsKeys = list(basicStats(final_year, fbsTeams[0]).keys())
advStatsKeys = list(advStats(final_year, fbsTeams[0]).keys())
metricsKeys = list(metrics(final_year, fbsTeams[0]).keys())
talentKeys = ['talent']
rpKeys = list(returningProd(final_year, fbsTeams[0]).keys())
fpiKeys = list(fpi(final_year, fbsTeams[0]).keys())
eloKeys = list(elo(final_year, fbsTeams[0]).keys())
coachKeys = list(coach(final_year, fbsTeams[0]).keys())

statsKeys = fpiKeys[3:] + eloKeys[3:] + baiscStatsKeys + advStatsKeys[3:] + metricsKeys[3:] + talentKeys + rpKeys[3:] + coachKeys[2:]
print(statsKeys)

savemat('teams.mat', {"teams": fbsTeams})
savemat('statsKeys.mat', {"statsKeys": statsKeys})

stats = np.zeros((len(fbsTeams), len(statsKeys), final_year - initial_year + 1))
missing = np.zeros((len(fbsTeams), final_year - initial_year + 1))
# winPctBySeason = np.zeros((len(fbsTeams), final_year - initial_year + 1))

# statLine += '_condensed'

for team in range(len(fbsTeams)):
    for year in range(initial_year, final_year+1):
        if year == 2020: continue
        bscStats = list(basicStats(year, fbsTeams[team]).values())
        advancedStats = list(advStats(year, fbsTeams[team]).values())
        metricsStats = list(metrics(year, fbsTeams[team]).values())
        talentStats = talent(year, fbsTeams[team])
        rpStats = list(returningProd(year, fbsTeams[team]).values())
        fpiStats = list(fpi(year, fbsTeams[team]).values())
        eloStats = list(elo(year, fbsTeams[team]).values())
        coachStats = list(coach(year, fbsTeams[team]).values())

        teamStats = [float(0 if x is None else x) for x in fpiStats[3:] + eloStats[3:] + bscStats + advancedStats[3:] + metricsStats[3:] + talentStats + rpStats[3:] + coachStats[2:]]

        if not all([any(bscStats), any(advancedStats), any(metricsStats), any(talentStats), any(rpStats), any(fpiStats), any(eloStats), any(coachStats)]):
            missing[team, year-initial_year] = 1

        stats[team, :, year-initial_year] = teamStats


savemat(statLine+'.mat', {statLine: stats})
savemat(statLine + '_teamsMissingStats.mat', {statLine + '_teamsMissingStats': missing})

stats = loadmat(statLine+'.mat')
stats = np.array(stats[statLine])

trackUnique = {}

gameInputs = []
gameOutputs = []

for team in range(len(fbsTeams)):
    for year in range(initial_year, final_year + 1):
        if year == 2020: continue
        for wk in range(13):

            gameStats = list(games(year, fbsTeams[team], wk+1).values())
            if gameStats == ['-1']: continue

            relStats = [13, 22, 1, 16, 25]   # home team, away team, season, home points, away points
            gameStatsCond = [gameStats[r] for r in relStats[0:2]] + [float(0 if gameStats[r] is None else gameStats[r]) for r in relStats[2:]]
            if gameStats[0] in trackUnique.keys(): continue
            print(gameStatsCond)

            trackUnique[gameStats[0]] = [gameStatsCond[0:1]]

            if gameStatsCond[0] not in fbsTeams or gameStatsCond[1] not in fbsTeams: continue

            homeID = fbsTeams.index(gameStatsCond[0])
            awayID = fbsTeams.index(gameStatsCond[1])

            homeStats = stats[homeID, :, gameStats[1] - initial_year]
            awayStats = stats[awayID, :, gameStats[1] - initial_year]

            # if all(homeStats == np.zeros_like(homeStats)) or all(awayStats == np.zeros_like(awayStats)):
            #     continue

            fullGame = np.concatenate((gameStatsCond[2:-2], homeStats, awayStats))
            outcome = gameStatsCond[-2] - gameStatsCond[-1]
            total = gameStatsCond[-2] + gameStatsCond[-1]

            gameInputs += [fullGame]
            # gameOutputs += [[outcome, total]]
            gameOutputs += [[gameStatsCond[-2], gameStatsCond[-1]]]

savemat("gameInputs_"+str(initial_year)+"_"+str(final_year)+"_full.mat", {"gameInputs": np.array(gameInputs)})
savemat("gameOutputs_"+str(initial_year)+"_"+str(final_year)+"_full.mat", {"gameOutputs": np.array(gameOutputs)})

# savemat("gameID.mat", trackUnique)
