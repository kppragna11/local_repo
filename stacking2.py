from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load the dataset
file_path = r'C:\Users\Student\Desktop\1ga21ec096\local_repo\silkboard.csv'
df = pd.read_csv(file_path,low_memory=False)

# Handle non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Select features and target variable
X = df_imputed.drop("PM2.5", axis=1)
y = df_imputed["PM2.5"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base regression models
linear_reg = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10, n_estimators=100)
svr_reg = SVR(kernel='linear', C=1.0)
knn_reg = KNeighborsRegressor(n_neighbors=5)
dt_reg = DecisionTreeRegressor(max_depth=10)

# Create a dictionary of models
models = {
    'Linear Regression': linear_reg,
    'Random Forest': rf_reg,
    'SVR': svr_reg,
    'K-Nearest Neighbors': knn_reg,
    'Decision Tree': dt_reg
}

# Perform grid search for each model
best_models = {}
for name, model in models.items():
    param_grid = {}  # You can add hyperparameter grids here
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_

# Create a stacking regressor without specifying the final_estimator
stacked_reg = StackingRegressor(estimators=list(best_models.items()))

# Perform grid search for the final_estimator with parallelization
final_estimator_param_dist = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_final_estimator = GradientBoostingRegressor(random_state=42)
grid_search_final = GridSearchCV(gb_final_estimator, param_grid=final_estimator_param_dist,
                                 scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_final.fit(X_train, y_train)

# Get the best hyperparameters for the final_estimator
best_final_estimator_params = grid_search_final.best_params_

# Set the final_estimator with the best hyperparameters
stacked_reg.final_estimator_ = GradientBoostingRegressor(**best_final_estimator_params)

# Fit the stacking regressor on the training data
stacked_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_stacked = stacked_reg.predict(X_test)

# Evaluate the performance of the stacking regressor
mse_stacked = mean_squared_error(y_test, y_pred_stacked)
rmse_stacked = np.sqrt(mse_stacked)
r2_stacked = r2_score(y_test, y_pred_stacked)

# Print the performance metrics
print("Mean Squared Error for Each Model:")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: {mse}")

print("\nBest Stacked Model:")
print(f"Stacked Regressor - MSE: {mse_stacked}, RMSE: {rmse_stacked}, R-squared: {r2_stacked}")
