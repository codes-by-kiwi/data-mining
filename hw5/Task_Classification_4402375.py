# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ucimlrepo import fetch_ucirepo 


wine_quality = fetch_ucirepo(id=186) 
  
X = wine_quality.data.features 
y = wine_quality.data.targets 

# Define the function for Random Forest classification with cross-validation
def random_forest_experiment(n_estimators, min_samples_leaf, X, y):

    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,random_state=2024)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=2024) 
    scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    return scores.mean()

# Function to return the best result based on hyperparameters
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    results = []

    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            accuracy = random_forest_experiment(n_estimators,min_samples_leaf,X,y)
            results.append({ 'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'accuracy': accuracy})

    best_result = max(results, key=lambda x: x['accuracy'])

    return best_result 

    # TO DO: After running the above function, manually input your best result here
def manually_entered_best_params_and_accuracy():
    best_params = {'n_estimators': 1000, 'min_samples_leaf': 1}  
    best_accuracy = 0.7096 
    return best_params, best_accuracy
    
# Main function
if __name__ == "__main__":

    n_estimators_list = [10, 50, 100, 1000]
    min_samples_leaf_list = [1, 10, 50, 100]

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)

    print(f"Best Hyperparameters: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best Accuracy: {best_params['accuracy']:.4f}")
