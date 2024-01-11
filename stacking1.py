from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Define base regression models with parallelization for RandomForest and GradientBoosting
linear_reg = LinearRegression()
lasso_reg = Lasso(alpha=0.1)
ridge_reg = Ridge(alpha=1.0)
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
huber_reg = HuberRegressor(epsilon=1.2)
svr_reg = SVR(kernel='linear', C=1.0)
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
knn_reg = KNeighborsRegressor(n_neighbors=5)
dt_reg = DecisionTreeRegressor(max_depth=10)

# Create a Stacking Regressor with a placeholder final_estimator
stacked_reg = StackingRegressor(
    estimators=[
        ('linear', linear_reg),
        ('lasso', lasso_reg),
        ('ridge', ridge_reg),
        ('elastic_net', elastic_net_reg),
        ('huber', huber_reg),
        ('svr', svr_reg),
        ('gradientboosting', gb_reg),
        ('knn', knn_reg),
        ('decisiontree', dt_reg)
    ],
    final_estimator=None
)

# Modify the potential_final_estimators dictionary
potential_final_estimators = {
    'Random Forest': {
        'model': RandomForestRegressor(),
        'param_dist': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'param_dist': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'param_dist': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(),
        'param_dist': {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
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
    'final_estimator__max_depth': [3, 5, 7],
    'final_estimator__n_estimators': [50, 100, 150],
    'linear__fit_intercept': [True, False],
    'lasso__alpha': [0.01, 0.1, 1.0],
    'ridge__alpha': [0.01, 0.1, 1.0],
    'elastic_net__alpha': [0.01, 0.1, 1.0],
    'elastic_net__l1_ratio': [0.1, 0.5, 0.9],
    'huber__epsilon': [1.1, 1.2, 1.3],
    'svr__C': [0.1, 1, 10],
    'gradientboosting__n_estimators': [50, 100, 150],
    'gradientboosting__max_depth': [3, 5, 7],
    'knn__n_neighbors': [3, 5, 7],
    'decisiontree__max_depth': [5, 10, 15]
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
print("Performance Metrics for Each Model:")
for name, mse, rmse, r2 in zip(model_names, mse_results.values(), rmse_results.values(), r2_results.values()):
    print(f"{name} - MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")

print("\nBest Stacked Model:")
print(f"Stacked Regressor - MSE: {mse_stacked}, RMSE: {rmse_stacked}, R-squared: {r2_stacked}")
