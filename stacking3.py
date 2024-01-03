from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Student\Desktop\1ga21ec096\local_repo\silkboard.csv'
df = pd.read_csv(file_path, low_memory=False)

# Handle non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Calculate z-scores for each column
z_scores = np.abs((df_imputed - df_imputed.mean()) / df_imputed.std())

# Remove rows with z-scores beyond a certain threshold (e.g., 3)
df_no_outliers = df_imputed[(z_scores < 3).all(axis=1)]

# Select features and target variable
X = df_no_outliers.drop("PM2.5", axis=1)
y = df_no_outliers["PM2.5"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Dictionary of base regression models with parallelization for RandomForest and GradientBoosting
linear_reg = LinearRegression()
lasso_reg = Lasso(alpha=0.1)
ridge_reg = Ridge(alpha=1.0)
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
huber_reg = HuberRegressor(epsilon=1.2)
svr_reg = SVR(kernel='linear', C=1.0)
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
knn_reg = KNeighborsRegressor(n_neighbors=5)
dt_reg = DecisionTreeRegressor(max_depth=10)

# More base models
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5)
extra_gb_reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4)
extra_knn_reg = KNeighborsRegressor(n_neighbors=10)

# Dictionary of base models
models = {
    'Linear Regression': linear_reg,
    'Lasso Regression': lasso_reg,
    'Ridge Regression': ridge_reg,
    'Elastic Net Regression': elastic_net_reg,
    'Huber Regression': huber_reg,
    'SVR': svr_reg,
    'Gradient Boosting': gb_reg,
    'KNN': knn_reg,
    'Decision Tree': dt_reg,
    'Random Forest': rf_reg,
    'Extra Gradient Boosting': extra_gb_reg,
    'Extra KNN': extra_knn_reg
}

# Create a Stacking Regressor with a placeholder final_estimator
stacked_reg = StackingRegressor(
    estimators=list(models.items()),
    final_estimator=None
)

# Dictionary to store cross-validation results
cv_results = {}

# Perform cross-validation for each base model
for name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_results[name] = np.sqrt(-cv_score)

# Display cross-validation results
print("Cross-Validation Results:")
for name, scores in cv_results.items():
    print(f"{name}: Mean RMSE = {np.mean(scores)}, Std Dev = {np.std(scores)}")

# Plot cross-validation results
plt.figure(figsize=(12, 6))
plt.boxplot(list(cv_results.values()), labels=list(cv_results.keys()))
plt.title('Cross-Validation Results')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.show()

# Modify the potential_final_estimators dictionary
potential_final_estimators = {
    'Random Forest': {
        'model': RandomForestRegressor(),
        'param_dist': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'param_dist': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'param_dist': {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(),
        'param_dist': {
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# Modify the final_estimator_param_dist dictionary
final_estimator_param_dist = {
    'final_estimator': list(potential_final_estimators.values())
}

# Perform a randomized search for the stacking regressor with parallelization
randomized_search = RandomizedSearchCV(
    stacked_reg,
    param_distributions=final_estimator_param_dist,
    n_iter=len(potential_final_estimators),  # Use the total number of final estimator models
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    error_score='raise'
)
randomized_search.fit(X_train, y_train)

# Get the best final estimator
best_final_estimator_config = randomized_search.best_params_['final_estimator']

# Use the best final estimator in the stacking regressor
stacked_reg.final_estimator = best_final_estimator_config['model']

# Define hyperparameter distributions for the stacking regressor
param_dist = {
    'final_estimator__max_depth': [3, 5],
    'final_estimator__n_estimators': [50, 100],
    'linear__fit_intercept': [True, False],
    'lasso__alpha': [0.01, 0.1],
    'ridge__alpha': [0.01, 0.1],
    'elastic_net__alpha': [0.01, 0.1],
    'elastic_net__l1_ratio': [0.1, 0.5],
    'huber__epsilon': [1.1, 1.2],
    'svr__C': [0.1, 1],
    'gradientboosting__n_estimators': [50, 100],
    'gradientboosting__max_depth': [3, 5],
    'knn__n_neighbors': [3, 5],
    'decisiontree__max_depth': [5, 10]
}

# Perform randomized search for the stacking regressor with parallelization
randomized_search_stacked = RandomizedSearchCV(
    stacked_reg,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)
randomized_search_stacked.fit(X_train, y_train)

# Get the best stacking regressor
best_stacked_model = randomized_search_stacked.best_estimator_

# Make predictions on the test set
y_pred_stacked = best_stacked_model.predict(X_test)

# Evaluate the performance of the best stacking regressor
mse_stacked = mean_squared_error(y_test, y_pred_stacked)
rmse_stacked = np.sqrt(mse_stacked)
r2_stacked = r2_score(y_test, y_pred_stacked)

# Print the performance metrics
print("\nBest Stacked Model:")
print(f"Stacked Regressor - MSE: {mse_stacked}, RMSE: {rmse_stacked}, R-squared: {r2_stacked}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_stacked, alpha=0.5)
plt.title("Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
