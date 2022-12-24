import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\kaggle世界杯数据集\results.csv")
df.head()

df.info()
print()
# 缺失值查看
print(df.isna().sum())

#删除缺失值所在的行
df = df.drop(df[df['date']=='2022-19-22'].index,axis=0)
df.dropna(inplace=True)
#将日期列的格式转换为日期格式
df["date"] = pd.to_datetime(df["date"])

df.sort_values("date").tail()
df = df[(df["date"] >= "2018-8-1")].reset_index(drop=True)
df.sort_values('date').tail()

rank = pd.read_csv(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\kaggle世界杯数据集\fifa_ranking-2022-10-06.csv")
rank["rank_date"] = pd.to_datetime(rank["rank_date"]) #转换日期格式
rank = rank[(rank["rank_date"] >= "2018-8-1")].reset_index(drop=True) #筛选数据集

rank["country_full"] = rank["country_full"].str.replace("IR Iran", "Iran").str\
    .replace("Korea Republic", "South Korea").str.replace("USA", "United States")

rank = rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()
df_wc_ranked = df.merge(rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
 left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"], axis=1)

df_wc_ranked = df_wc_ranked.merge(rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
left_on=["date", "away_team"], right_on=["rank_date", "country_full"],
suffixes=("_home", "_away")).drop(["rank_date", "country_full"], axis=1)

print(df_wc_ranked[(df_wc_ranked.home_team == "Germany") | (df_wc_ranked.away_team == "Germany")].tail())

# 判断函数
df = df_wc_ranked
def result_finder(home, away):
    if home > away:
        return pd.Series([0, 3, 0])
    if home < away:
        return pd.Series([1, 0, 3])
    else:
        return pd.Series([2, 1, 1])

results = df.apply(lambda x: result_finder(x["home_score"], x["away_score"]), axis=1)
df[["result", "home_team_points", "away_team_points"]] = results
# 假设检验
plt.figure(figsize=(15, 10))
sns.heatmap(df[["total_points_home", "rank_home", "total_points_away", "rank_away"]].corr())
# plt.show()

df["rank_dif"] = df["rank_home"] - df["rank_away"]  #排名差异
df["sg"] = df["home_score"] - df["away_score"]  #分数差异
df["points_home_by_rank"] = df["home_team_points"]/df["rank_away"] #主场队伍进球与排名的关系
df["points_away_by_rank"] = df["away_team_points"]/df["rank_home"] #客场队伍进球与排名的关系

home_team = df[["date", "home_team", "home_score", "away_score", "rank_home", "rank_away","rank_change_home", "total_points_home", "result", "rank_dif", "points_home_by_rank", "home_team_points"]]
away_team = df[["date", "away_team", "away_score", "home_score", "rank_away", "rank_home","rank_change_away", "total_points_away", "result", "rank_dif", "points_away_by_rank", "away_team_points"]]
home_team.columns = [h.replace("home_", "").replace("_home", "").replace("away_", "suf_").replace("_away", "_suf") for h in home_team.columns]
away_team.columns = [a.replace("away_", "").replace("_away", "").replace("home_", "suf_").replace("_home", "_suf") for a in away_team.columns]

team_stats = home_team.append(away_team)
team_stats_raw = team_stats.copy()

stats_val = []

for index, row in team_stats.iterrows():
    team = row["team"]
    date = row["date"]
    past_games = team_stats.loc[(team_stats["team"] == team) & (team_stats["date"] < date)].sort_values(by=['date'],
                                                                                                        ascending=False)
    last5 = past_games.head(5)  # 取出过去五场比赛

    goals = past_games["score"].mean()
    goals_l5 = last5["score"].mean()

    goals_suf = past_games["suf_score"].mean()
    goals_suf_l5 = last5["suf_score"].mean()

    rank = past_games["rank_suf"].mean()
    rank_l5 = last5["rank_suf"].mean()

    if len(last5) > 0:
        points = past_games["total_points"].values[0] - past_games["total_points"].values[-1]  # qtd de pontos ganhos
        points_l5 = last5["total_points"].values[0] - last5["total_points"].values[-1]
    else:
        points = 0
        points_l5 = 0

    gp = past_games["team_points"].mean()
    gp_l5 = last5["team_points"].mean()

    gp_rank = past_games["points_by_rank"].mean()
    gp_rank_l5 = last5["points_by_rank"].mean()

    stats_val.append([goals, goals_l5, goals_suf, goals_suf_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank, gp_rank_l5])

stats_cols = ["goals_mean", "goals_mean_l5", "goals_suf_mean", "goals_suf_mean_l5", "rank_mean", "rank_mean_l5",
                  "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5", "game_points_rank_mean",
                  "game_points_rank_mean_l5"]

stats_df = pd.DataFrame(stats_val, columns=stats_cols)

full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)

home_team_stats = full_df.iloc[:int(full_df.shape[0] / 2), :]
away_team_stats = full_df.iloc[int(full_df.shape[0] / 2):, :]

home_team_stats = home_team_stats[home_team_stats.columns[-12:]]
away_team_stats = away_team_stats[away_team_stats.columns[-12:]]

home_team_stats.columns = ['home_' + str(col) for col in home_team_stats.columns]
away_team_stats.columns = ['away_' + str(col) for col in away_team_stats.columns]

match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)
full_df = pd.concat([df, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)

print(full_df.columns)

def find_friendly(x):
    if x == "Friendly":
        return 1
    else: return 0

full_df["is_friendly"] = full_df["tournament"].apply(lambda x: find_friendly(x))
full_df = pd.get_dummies(full_df, columns=["is_friendly"])

base_df = full_df[["date", "home_team", "away_team", "rank_home", "rank_away","home_score", "away_score","result", "rank_dif", "rank_change_home", "rank_change_away", 'home_goals_mean',
       'home_goals_mean_l5', 'home_goals_suf_mean', 'home_goals_suf_mean_l5',
       'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean',
       'home_points_mean_l5', 'away_goals_mean', 'away_goals_mean_l5',
       'away_goals_suf_mean', 'away_goals_suf_mean_l5', 'away_rank_mean',
       'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5','home_game_points_mean', 'home_game_points_mean_l5',
       'home_game_points_rank_mean', 'home_game_points_rank_mean_l5','away_game_points_mean',
       'away_game_points_mean_l5', 'away_game_points_rank_mean',
       'away_game_points_rank_mean_l5',
       'is_friendly_0', 'is_friendly_1']]

print(base_df.head())
print(base_df.isna().sum())

base_df_no_fg = base_df.dropna()
df = base_df_no_fg

def no_draw(x):
    if x == 2:
        return 1
    else:
        return x

df["target"] = df["result"].apply(lambda x: no_draw(x))

data1 = df[list(df.columns[8:20].values) + ["target"]]
data2 = df[df.columns[20:]]

scaled = (data1[:-1] - data1[:-1].mean()) / data1[:-1].std()
scaled["target"] = data1["target"]
violin1 = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")

scaled = (data2[:-1] - data2[:-1].mean()) / data2[:-1].std()
scaled["target"] = data2["target"]
violin2 = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")

plt.figure(figsize=(15,10))
sns.violinplot(x="features", y="value", hue="target", data=violin1,split=True, inner="quart")
plt.xticks(rotation=90)
# plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(x="features", y="value", hue="target", data=violin2,split=True, inner="quart")
plt.xticks(rotation=90)
# plt.show()

dif = df.copy()
dif.loc[:, "goals_dif"] = dif["home_goals_mean"] - dif["away_goals_mean"]
dif.loc[:, "goals_dif_l5"] = dif["home_goals_mean_l5"] - dif["away_goals_mean_l5"]
dif.loc[:, "goals_suf_dif"] = dif["home_goals_suf_mean"] - dif["away_goals_suf_mean"]
dif.loc[:, "goals_suf_dif_l5"] = dif["home_goals_suf_mean_l5"] - dif["away_goals_suf_mean_l5"]
dif.loc[:, "goals_made_suf_dif"] = dif["home_goals_mean"] - dif["away_goals_suf_mean"]
dif.loc[:, "goals_made_suf_dif_l5"] = dif["home_goals_mean_l5"] - dif["away_goals_suf_mean_l5"]
dif.loc[:, "goals_suf_made_dif"] = dif["home_goals_suf_mean"] - dif["away_goals_mean"]
dif.loc[:, "goals_suf_made_dif_l5"] = dif["home_goals_suf_mean_l5"] - dif["away_goals_mean_l5"]

data_difs = dif.iloc[:, -8:]
scaled = (data_difs - data_difs.mean()) / data_difs.std()
scaled["target"] = data2["target"]
violin = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")

plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="target", data=violin,split=True, inner="quart")
plt.xticks(rotation=90)
# plt.show()

dif.loc[:, "dif_points"] = dif["home_game_points_mean"] - dif["away_game_points_mean"]
dif.loc[:, "dif_points_l5"] = dif["home_game_points_mean_l5"] - dif["away_game_points_mean_l5"]
dif.loc[:, "dif_points_rank"] = dif["home_game_points_rank_mean"] - dif["away_game_points_rank_mean"]
dif.loc[:, "dif_points_rank_l5"] = dif["home_game_points_rank_mean_l5"] - dif["away_game_points_rank_mean_l5"]

dif.loc[:, "dif_rank_agst"] = dif["home_rank_mean"] - dif["away_rank_mean"]
dif.loc[:, "dif_rank_agst_l5"] = dif["home_rank_mean_l5"] - dif["away_rank_mean_l5"]

dif.loc[:, "goals_per_ranking_dif"] = (dif["home_goals_mean"] / dif["home_rank_mean"]) - (dif["away_goals_mean"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_suf_dif"] = (dif["home_goals_suf_mean"] / dif["home_rank_mean"]) - (dif["away_goals_suf_mean"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_dif_l5"] = (dif["home_goals_mean_l5"] / dif["home_rank_mean"]) - (dif["away_goals_mean_l5"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_suf_dif_l5"] = (dif["home_goals_suf_mean_l5"] / dif["home_rank_mean"]) - (dif["away_goals_suf_mean_l5"] / dif["away_rank_mean"])

data_difs = dif.iloc[:, -10:]
scaled = (data_difs - data_difs.mean()) / data_difs.std()
scaled["target"] = data2["target"]
violin = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")

plt.figure(figsize=(15,10))
sns.violinplot(x="features", y="value", hue="target", data=violin,split=True, inner="quart")
plt.xticks(rotation=90)
# plt.show()

plt.figure(figsize=(15,10))
sns.boxplot(x="features", y="value", hue="target", data=violin)
plt.xticks(rotation=90)
# plt.show()

sns.jointplot(data = data_difs, x = 'goals_per_ranking_dif', y = 'goals_per_ranking_dif_l5', kind="reg")
# plt.show()

sns.jointplot(data = data_difs, x = 'dif_rank_agst', y = 'dif_rank_agst_l5', kind="reg")
# plt.show()

sns.jointplot(data = data_difs, x = 'dif_points', y = 'dif_points_l5', kind="reg")
# plt.show()

sns.jointplot(data = data_difs, x = 'dif_points_rank', y = 'dif_points_rank_l5', kind="reg")
# plt.show()


def create_db(df):
    columns = ["home_team", "away_team", "target", "rank_dif", "home_goals_mean", "home_rank_mean", "away_goals_mean",
               "away_rank_mean", "home_rank_mean_l5", "away_rank_mean_l5", "home_goals_suf_mean", "away_goals_suf_mean",
               "home_goals_mean_l5", "away_goals_mean_l5", "home_goals_suf_mean_l5", "away_goals_suf_mean_l5",
               "home_game_points_rank_mean", "home_game_points_rank_mean_l5", "away_game_points_rank_mean",
               "away_game_points_rank_mean_l5", "is_friendly_0", "is_friendly_1"]

    base = df.loc[:, columns]
    base.loc[:, "goals_dif"] = base["home_goals_mean"] - base["away_goals_mean"]
    base.loc[:, "goals_dif_l5"] = base["home_goals_mean_l5"] - base["away_goals_mean_l5"]
    base.loc[:, "goals_suf_dif"] = base["home_goals_suf_mean"] - base["away_goals_suf_mean"]
    base.loc[:, "goals_suf_dif_l5"] = base["home_goals_suf_mean_l5"] - base["away_goals_suf_mean_l5"]
    base.loc[:, "goals_per_ranking_dif"] = (base["home_goals_mean"] / base["home_rank_mean"]) - (
                base["away_goals_mean"] / base["away_rank_mean"])
    base.loc[:, "dif_rank_agst"] = base["home_rank_mean"] - base["away_rank_mean"]
    base.loc[:, "dif_rank_agst_l5"] = base["home_rank_mean_l5"] - base["away_rank_mean_l5"]
    base.loc[:, "dif_points_rank"] = base["home_game_points_rank_mean"] - base["away_game_points_rank_mean"]
    base.loc[:, "dif_points_rank_l5"] = base["home_game_points_rank_mean_l5"] - base["away_game_points_rank_mean_l5"]

    model_df = base[["home_team", "away_team", "target", "rank_dif", "goals_dif", "goals_dif_l5", "goals_suf_dif",
                     "goals_suf_dif_l5", "goals_per_ranking_dif", "dif_rank_agst", "dif_rank_agst_l5",
                     "dif_points_rank", "dif_points_rank_l5", "is_friendly_0", "is_friendly_1"]]
    return model_df


model_db = create_db(df)
print(model_db)

X = model_db.iloc[:, 3:]
y = model_db[["target"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

# GDBT
gb = GradientBoostingClassifier(random_state=5)

params = {"learning_rate": [0.01, 0.1, 0.5],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [3, 5],
            "max_depth":[3,5,10],
            "max_features":["sqrt"],
            "n_estimators":[100, 200]
         }

gb_cv = GridSearchCV(gb, params, cv = 3, n_jobs = -1, verbose = False)
gb_cv.fit(X_train.values, np.ravel(y_train))
print(gb_cv)

gb = gb_cv.best_estimator_
print(gb)

# RFC
params_rf = {"max_depth": [20],
                "min_samples_split": [10],
                "max_leaf_nodes": [175],
                "min_samples_leaf": [5],
                "n_estimators": [250],
                 "max_features": ["sqrt"],
                }

rf = RandomForestClassifier(random_state=1)
rf_cv = GridSearchCV(rf, params_rf, cv = 3, n_jobs = -1, verbose = False)
rf_cv.fit(X_train.values, np.ravel(y_train))
rf = rf_cv.best_estimator_
print(rf)


def analyze(model):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test.values)[:, 1])  # test AUC
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="test")

    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train.values)[:, 1])  # train AUC
    plt.plot(fpr_train, tpr_train, label="train")
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train.values)[:, 1])
    plt.legend()
    plt.title('AUC score is %.2f on test and %.2f on training' % (auc_test, auc_train))
    plt.show()

    plt.figure(figsize=(15, 10))
    cm = confusion_matrix(y_test, model.predict(X_test.values))
    sns.heatmap(cm, annot=True, fmt="d")

# analyze(gb)
# analyze(rf)

from operator import itemgetter
'''
dfs = pd.read_html(r"https://en.wikipedia.org/wiki/2022_FIFA_World_Cup#Teams")

from collections.abc import Iterable

for i in range(len(dfs)):
    df = dfs[i]
    cols = list(df.columns.values)

    if isinstance(cols[0], Iterable):
        if any("Tie-breaking criteria" in c for c in cols):
            start_pos = i + 1

        if any("Match 46" in c for c in cols):
            end_pos = i + 1

matches = []
groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
group_count = 0

table = {}
table[groups[group_count]] = [[a.split(" ")[0], 0, []] for a in list(dfs[start_pos].iloc[:, 1].values)]

for i in range(start_pos + 1, end_pos, 1):
    if len(dfs[i].columns) == 3:
        team_1 = dfs[i].columns.values[0]
        team_2 = dfs[i].columns.values[-1]

        matches.append((groups[group_count], team_1, team_2))
    else:
        group_count += 1
        table[groups[group_count]] = [[a, 0, []] for a in list(dfs[i].iloc[:, 1].values)]

print(table)
'''
table = {
    'A':[['Qatar',0,[]],['Ecuador',0,[]],['Senegal',0,[]],['Netherlands',0,[]]],
'B':[['England',0,[]],['Iran',0,[]],['United States',0,[]],['Wales',0,[]]],
'C':[['Argentina',0,[]],['Saudi Arabia',0,[]],['Mexico',0,[]],['Poland',0,[]]],
'D':[['France',0,[]],['Australia',0,[]],['Denmark',0,[]],['Tunisia',0,[]]],
'E':[['Spain',0,[]],['Costa Rica',0,[]],['Germany',0,[]],['Japan',0,[]]],
'F':[['Belgium',0,[]],['Canada',0,[]],['Morocco',0,[]],['Croatia',0,[]]],
'G':[['Brazil',0,[]],['Serbia',0,[]],['Switzerland',0,[]],['Cameroon',0,[]]],
'H':[['Portugal',0,[]],['Ghana',0,[]],['Uruguay',0,[]],['South Korea',0,[]]]
}

matches = [['A','Qatar','Ecuador'],['A','Qatar','Senegal'],['A','Qatar','Netherlands'],['A','Ecuador','Senegal'],['A','Ecuador','Netherlands'],['A','Senegal','Netherlands'],
['B','England','Iran'],['B','England','United States'],['B','England','Wales'],['B','Iran','United States'],['B','Iran','Wales'],['B','United States','Wales'],
['C','Argentina','Saudi Arabia'],['C','Argentina','Mexico'],['C','Argentina','Poland'],['C','Saudi Arabia','Mexico'],['C','Saudi Arabia','Poland'],['C','Mexico','Poland'],
['D','France','Australia'],['D','France','Denmark'],['D','France','Tunisia'],['D','Australia','Denmark'],['D','Australia','Tunisia'],['D','Denmark','Tunisia'],
['E','Spain','Costa Rica'],['E','Spain','Germany'],['E','Spain','Japan'],['E','Costa Rica','Germany'],['E','Costa Rica','Japan'],['E','Germany','Japan'],
['F','Belgium','Canada'],['F','Belgium','Morocco'],['F','Belgium','Croatia'],['F','Canada','Morocco'],['F','Canada','Croatia'],['F','Morocco','Croatia'],
['G','Brazil','Serbia'],['G','Brazil','Switzerland'],['G','Brazil','Cameroon'],['G','Serbia','Switzerland'],['G','Serbia','Cameroon'],['G','Switzerland','Cameroon'],
['H','Portugal','Ghana'],['H','Portugal','Uruguay'],['H','Portugal','South Korea'],['H','Ghana','Uruguay'],['H','Ghana','South Korea'],['H','Uruguay','South Korea']
]

def find_stats(team_1):
    # team_1 = "Qatar"
    past_games = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date")
    last5 = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date").tail(5)

    team_1_rank = past_games["rank"].values[-1]
    team_1_goals = past_games.score.mean()
    team_1_goals_l5 = last5.score.mean()
    team_1_goals_suf = past_games.suf_score.mean()
    team_1_goals_suf_l5 = last5.suf_score.mean()
    team_1_rank_suf = past_games.rank_suf.mean()
    team_1_rank_suf_l5 = last5.rank_suf.mean()
    team_1_gp_rank = past_games.points_by_rank.mean()
    team_1_gp_rank_l5 = last5.points_by_rank.mean()

    return [team_1_rank, team_1_goals, team_1_goals_l5, team_1_goals_suf, team_1_goals_suf_l5, team_1_rank_suf,
            team_1_rank_suf_l5, team_1_gp_rank, team_1_gp_rank_l5]


def find_features(team_1, team_2):
    rank_dif = team_1[0] - team_2[0]
    goals_dif = team_1[1] - team_2[1]
    goals_dif_l5 = team_1[2] - team_2[2]
    goals_suf_dif = team_1[3] - team_2[3]
    goals_suf_dif_l5 = team_1[4] - team_2[4]
    goals_per_ranking_dif = (team_1[1] / team_1[5]) - (team_2[1] / team_2[5])
    dif_rank_agst = team_1[5] - team_2[5]
    dif_rank_agst_l5 = team_1[6] - team_2[6]
    dif_gp_rank = team_1[7] - team_2[7]
    dif_gp_rank_l5 = team_1[8] - team_2[8]

    return [rank_dif, goals_dif, goals_dif_l5, goals_suf_dif, goals_suf_dif_l5, goals_per_ranking_dif, dif_rank_agst,
            dif_rank_agst_l5, dif_gp_rank, dif_gp_rank_l5, 1, 0]


advanced_group = []
last_group = ""

for k in table.keys():
    for t in table[k]:
        t[1] = 0
        t[2] = []

for teams in matches:
    draw = False
    team_1 = find_stats(teams[1])
    team_2 = find_stats(teams[2])

    features_g1 = find_features(team_1, team_2)
    features_g2 = find_features(team_2, team_1)

    probs_g1 = gb.predict_proba([features_g1])
    probs_g2 = gb.predict_proba([features_g2])

    team_1_prob_g1 = probs_g1[0][0]
    team_1_prob_g2 = probs_g2[0][1]
    team_2_prob_g1 = probs_g1[0][1]
    team_2_prob_g2 = probs_g2[0][0]

    team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
    team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

    if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | (
            (team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
        draw = True
        for i in table[teams[0]]:
            if i[0] == teams[1] or i[0] == teams[2]:
                i[1] += 1

    elif team_1_prob > team_2_prob:
        winner = teams[1]
        winner_proba = team_1_prob
        for i in table[teams[0]]:
            if i[0] == teams[1]:
                i[1] += 3

    elif team_2_prob > team_1_prob:
        winner = teams[2]
        winner_proba = team_2_prob
        for i in table[teams[0]]:
            if i[0] == teams[2]:
                i[1] += 3

    for i in table[teams[0]]:  # adding criterio de desempate (probs por jogo)
        if i[0] == teams[1]:
            i[2].append(team_1_prob)
        if i[0] == teams[2]:
            i[2].append(team_2_prob)

    if last_group != teams[0]:
        if last_group != "":
            print("\n")
            print("%s组 : " % (last_group))

            for i in table[last_group]:  # adding crieterio de desempate
                i[2] = np.mean(i[2])

            final_points = table[last_group]
            final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
            advanced_group.append([final_table[0][0], final_table[1][0]])
            for i in final_table:
                print("%s -------- %d" % (i[0], i[1]))
        print("\n")
        print("-" * 10 + "  %s组开始分析 " % (teams[0]) + "-" * 10)

    if draw == False:
        print(" %s组 - %s VS. %s:  %s获胜 概率为 %.2f" % (teams[0], teams[1], teams[2], winner, winner_proba))
    else:
        print(" %s组 - %s vs. %s: 平局" % (teams[0], teams[1], teams[2]))
    last_group = teams[0]

print("\n")
print(" %s组 : " % (last_group))

for i in table[last_group]:  # adding crieterio de desempate
    i[2] = np.mean(i[2])

final_points = table[last_group]
final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
advanced_group.append([final_table[0][0], final_table[1][0]])
for i in final_table:
    print("%s -------- %d" % (i[0], i[1]))

advanced = [['Netherlands','Senegal'],['England','United States'],['Argentina','Poland'],['France','Australia'],
            ['Japan','Spain'],['Morocco','Croatia'],['Brazil','Switzerland'],['Portugal','South Korea']]


playoffs = {"八分之一决赛": [], "四分之一决赛": [], "半决赛": [], "决赛": []}

for p in playoffs.keys():
    playoffs[p] = []

actual_round = ""
next_rounds = []

for p in playoffs.keys():
    if p == "八分之一决赛":
        control = []
        for a in range(0, len(advanced * 2), 1):
            if a < len(advanced):
                if a % 2 == 0:
                    control.append((advanced * 2)[a][0])
                else:
                    control.append((advanced * 2)[a][1])
            else:
                if a % 2 == 0:
                    control.append((advanced * 2)[a][1])
                else:
                    control.append((advanced * 2)[a][0])

        playoffs[p] = [[control[c], control[c + 1]] for c in range(0, len(control) - 1, 1) if c % 2 == 0]

        for i in range(0, len(playoffs[p]), 1):
            game = playoffs[p][i]

            home = game[0]
            away = game[1]
            team_1 = find_stats(home)
            team_2 = find_stats(away)

            features_g1 = find_features(team_1, team_2)
            features_g2 = find_features(team_2, team_1)

            probs_g1 = gb.predict_proba([features_g1])
            probs_g2 = gb.predict_proba([features_g2])

            team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
            team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

            if actual_round != p:
                print("-" * 10)
                print("开始模拟 %s" % (p))
                print("-" * 10)
                print("\n")

            if team_1_prob < team_2_prob:
                print("%s VS. %s: %s 晋级 概率为 %.2f" % (home, away, away, team_2_prob))
                next_rounds.append(away)
            else:
                print("%s VS. %s: %s 晋级 概率为 %.2f" % (home, away, home, team_1_prob))
                next_rounds.append(home)

            game.append([team_1_prob, team_2_prob])
            playoffs[p][i] = game
            actual_round = p

    else:
        playoffs[p] = [[next_rounds[c], next_rounds[c + 1]] for c in range(0, len(next_rounds) - 1, 1) if c % 2 == 0]
        next_rounds = []
        for i in range(0, len(playoffs[p])):
            game = playoffs[p][i]
            home = game[0]
            away = game[1]
            team_1 = find_stats(home)
            team_2 = find_stats(away)

            features_g1 = find_features(team_1, team_2)
            features_g2 = find_features(team_2, team_1)

            probs_g1 = gb.predict_proba([features_g1])
            probs_g2 = gb.predict_proba([features_g2])

            team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
            team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

            if actual_round != p:
                print("-" * 10)
                print("开始模拟 %s" % (p))
                print("-" * 10)
                print("\n")

            if team_1_prob < team_2_prob:
                print("%s VS. %s: %s 晋级 概率为 %.2f" % (home, away, away, team_2_prob))
                next_rounds.append(away)
            else:
                print("%s VS. %s: %s 晋级 概率为 %.2f" % (home, away, home, team_1_prob))
                next_rounds.append(home)
            game.append([team_1_prob, team_2_prob])
            playoffs[p][i] = game
            actual_round = p

'''
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot

plt.figure(figsize=(15, 10))
G = nx.balanced_tree(2, 3)

labels = []

for p in playoffs.keys():
    for game in playoffs[p]:
        label = f"{game[0]}({round(game[2][0], 2)}) \n {game[1]}({round(game[2][1], 2)})"
        labels.append(label)

labels_dict = {}
labels_rev = list(reversed(labels))

for l in range(len(list(G.nodes))):
    labels_dict[l] = labels_rev[l]

pos = graphviz_layout(G, prog='twopi')
labels_pos = {n: (k[0], k[1] - 0.08 * k[1]) for n, k in pos.items()}
center = pd.DataFrame(pos).mean(axis=1).mean()

nx.draw(G, pos=pos, with_labels=False, node_color=range(15), edge_color="#66CCFF", width=10, font_weight='bold', node_size=5000)
nx.draw_networkx_labels(G, pos=labels_pos, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1),
                        labels=labels_dict)
texts = ["1/16 \n 决赛", "1/8 \n 决赛", "半决赛\n", "决赛\n"]
pos_y = pos[0][1] + 55
for text in reversed(texts):
    pos_x = center
    pos_y -= 75
    plt.text(pos_y, pos_x, text, fontsize=18)

plt.axis('equal')
plt.show()
'''

