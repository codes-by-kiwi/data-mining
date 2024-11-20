import pandas as pd
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Data loading and preprocessing functions
def load_and_preprocess_data():
    """
    Load and preprocess the bike sharing dataset. This dataset includes dates and other features related to the number of bike rentals.
    Returns the feature matrix X and the target variable y.
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

    try:
        zip_file_loc = "Bike-Data.zip"
    
        download_response = requests.get(data_url)
        with open(zip_file_loc, 'wb') as f:
            f.write(download_response.content)
    
        with zipfile.ZipFile(zip_file_loc, 'r') as zip_file:
            zip_file.extractall("Bike-Data")
    
        data = pd.read_csv("Bike-Data/day.csv")

        data['dteday'] = pd.to_datetime(data['dteday'])
        data['day_of_month'] = data['dteday'].dt.day
        y = data['cnt']
        removable_cols = ['dteday', 'casual', 'registered', 'cnt']
        data.drop(columns=removable_cols, inplace=True)

        X = data 

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or reading csv the file: {e}")
    
    return X, y

# Define the function for the random forest regression experiment
def random_forest_regression_experiment(n_estimators, min_samples_leaf, X, y):

    
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=42)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=42) 
    meansqerror_scores = cross_val_score(model, X, y, cv=k_fold, scoring='neg_mean_squared_error')

    #mse_scores = -meansqerror_scores
    total_mse = sum(meansqerror_scores)
    mse = total_mse / len(meansqerror_scores)
    return mse

# Function to find the best hyperparameter combination
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    results = []

    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            mse = random_forest_regression_experiment(n_estimators, min_samples_leaf, X, y)
            results.append({ 'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'mse': mse})
    
    best_result = min(results, key=lambda x: abs(x['mse']))
    return best_result # best_result is a dictionary


    # TO DO: After running the above function, manually input your best result here
def manually_entered_best_params_and_mse():
    best_params = {'n_estimators': 1000, 'min_samples_leaf': 1}  # Example, to be replaced with the retrieved best parameters
    best_mse = -418641.1448  # Example mse, to be replaced with your achieved mse
    return best_params, best_mse

if __name__ == "__main__":

    X, y = load_and_preprocess_data()
    n_estimators_list = [10, 50, 100, 1000] 
    min_samples_leaf_list = [1, 10, 50, 100] 

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)
    print(f"Best Hyperparameter: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best negative mean square error: {best_params['mse']:.4f}")

