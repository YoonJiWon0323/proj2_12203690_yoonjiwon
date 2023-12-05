import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 2-1
data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

np.random.seed(0)
years = [2015, 2016, 2017, 2018]
positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
columns = ['batter_name', 'H', 'avg', 'HR', 'OBP', 'R', 'RBI', 'SB', 'war', 'SLG', 'salary']

players = []
for year in years:
    for position in positions:
        for num in range(15):
            batter_name = f'Player' 
            player_info = {
                'year': year,
                'position': position,
                'player': f'{batter_name}: {year}_{num}'
            }
            for column in columns:
                player_info[column] = np.random.rand() if column != 'salary' else np.random.randint(10000, 50000)
            players.append(player_info)

player_df = pd.DataFrame(players)

def top_players(dataframe, year, metric):
    top10 = dataframe[dataframe['year'] == year].sort_values(by=metric, ascending=False).head(10)
    print(f"Top 10 in {metric}, {year}")
    for idx, row in top10.iterrows():
        print(f"{idx + 1}. {row['player']}\n")

for y in years:
    for m in ['H', 'avg', 'HR', 'OBP']:
        top_players(player_df, y, m)

def best_war_player(dataframe, year):
    print(f"Best WAR Players in {year}")
    for position in positions:
        best = dataframe[(dataframe['year'] == year) & (dataframe['position'] == position)].sort_values(by='war', ascending=False).iloc[0]
        print(f"{position}: {best['player']} -> WAR: {best['war']}\n")

best_war_player(player_df, 2018)

def salary_correlation(dataframe):
    relevant_cols = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']
    correlations = dataframe[relevant_cols].corr()['salary']
    highest_corr_col = correlations.drop('salary').idxmax()
    print(f"Highest correlation is with salary: {highest_corr_col}\n")

salary_correlation(player_df)

# 2-2
def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year')


def split_dataset(dataset_df):
    X = dataset_df.drop(columns=['salary'])
    Y = dataset_df['salary'] * 0.001
    return X.iloc[:1718], X.iloc[1718:], Y.iloc[:1718], Y.iloc[1718:]


def extract_numerical_cols(dataset_df):
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    return dataset_df[numerical_cols]


def train_predict_decision_tree(X_train, Y_train, X_test):
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, Y_train)
    return decision_tree.predict(X_test)


def train_predict_random_forest(X_train, Y_train, X_test):
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, Y_train)
    return random_forest.predict(X_test)


def train_predict_svm(X_train, Y_train, X_test):
    svm = make_pipeline(StandardScaler(), SVR())
    svm.fit(X_train, Y_train)
    return svm.predict(X_test)


def calculate_RMSE(labels, predictions):
    return np.sqrt(mean_squared_error(labels, predictions))


if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.0
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))