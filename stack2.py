import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
file_path=(r'C:\Users\Student\Desktop\1ga21ec096\local_repo\silkboard.csv')
df = pd.read_csv(file_path,low_memory=False)

# Handle non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Display the first few rows of the dataset
columns = ['PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone', 'Benzene', 'Toluene', 'Temp', 'RH', 'WS', 'WD', 'BP']
df_imputed.head()
# Define X (features) and y (target variable)
X = df_imputed[columns]
y = df_imputed['PM2.5']
 #Define parameters for models
params_xgb = {'lambda': 0.7044156083795233, 'alpha': 9.681476940192473, 'colsample_bytree': 0.3, 'subsample': 0.8,
              'learning_rate': 0.015, 'max_depth': 3, 'min_child_weight': 235, 'random_state': 48, 'n_estimators': 30000}

params_lgb = {'reg_alpha': 4.973064761998367, 'reg_lambda': 0.06365096912006087, 'colsample_bytree': 0.24,
              'subsample': 0.8, 'learning_rate': 0.015, 'max_depth': 100, 'num_leaves': 43, 'min_child_samples': 141,
              'cat_smooth': 18, 'metric': 'rmse', 'random_state': 48, 'n_estimators': 40000}

params_rf= {
            'n_estimators': 800,
            'max_depth': 5,
            'min_samples_split': 3,
            'min_samples_leaf': 2}
params_gb={
       
            'n_estimators': 800,
            'max_depth': 5,
            'learning_rate': 0.01}
params_knn={'n_neighbors': 3}
params_dt= {
       
            'max_depth': 5, 
            'min_samples_split': 5,
            'min_samples_leaf': 2}
# Initialize arrays for predictions
pred1 = np.zeros(df_imputed.shape[0])
pred2 = np.zeros(df_imputed.shape[0])
pred3 = np.zeros(df_imputed.shape[0])
pred4 = np.zeros(df_imputed.shape[0])
pred5 = np.zeros(df_imputed.shape[0])
pred6 = np.zeros(df_imputed.shape[0])
pred7 = np.zeros(df_imputed.shape[0])
pred8 = np.zeros(df_imputed.shape[0])
pred9 = np.zeros(df_imputed.shape[0])
pred10 = np.zeros(df_imputed.shape[0])
pred11= np.zeros(df_imputed.shape[0])


kf = KFold(n_splits=10, random_state=48, shuffle=True)
n = 0
for trn_idx, test_idx in kf.split(X, y):
    print(f"fold: {n+1}")
    X_tr, X_val = X.iloc[trn_idx], X.iloc[test_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[test_idx]
        # Model 1: LGBMRegressor
    model1 = lgb.LGBMRegressor(**params_lgb)
    model1.fit(X_tr, y_tr)  # Remove the early_stopping_rounds parameter
    pred1[test_idx] = model1.predict(X_val)
 
    # Model 2: ElasticNet
    model2 = ElasticNet(alpha=0.00001, max_iter=10000)

    model2.fit(X_tr, y_tr)
    pred2[test_idx] = model2.predict(X_val)

    # Model 3: LinearRegression
    model3 = LinearRegression()
    model3.fit(X_tr, y_tr)
    pred3[test_idx] = model3.predict(X_val)

    # Model 4: XGBRegressor
    model4 = xgb.XGBRegressor(**params_xgb)
    model4.set_params(early_stopping_rounds=200)  # You can set other parameters if needed
    model4.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    pred4[test_idx] = model4.predict(X_val)

    model5 = Ridge(alpha=1.0)
    model5.fit(X_tr, y_tr)
    pred5[test_idx] = model5.predict(X_val)
    
    model6= RandomForestRegressor(**params_rf)
    model6.fit(X_tr, y_tr)
    pred6[test_idx] = model6.predict(X_val)
    
    model7= HuberRegressor(epsilon=1.2,max_iter=1000)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    model7 = make_pipeline(scaler, model7)
    model7.fit(X_tr, y_tr)
    pred7[test_idx] = model7.predict(X_val)
    
    
    model9 = GradientBoostingRegressor(**params_gb)
    model9.fit(X_tr, y_tr)
    pred9[test_idx] = model9.predict(X_val)
    
    model10 = KNeighborsRegressor(**params_knn)
    model10.fit(X_tr, y_tr)
    pred10[test_idx] = model10.predict(X_val)
    
    model11= DecisionTreeRegressor(**params_dt)
    model11.fit(X_tr, y_tr)
    pred11[test_idx] = model11.predict(X_val)

    n += 1
    # Stack the predictions
stacked_predictions = np.column_stack((pred1, pred2, pred3, pred4,pred5, pred6, pred7,pred9, pred10, pred11))
# Define the meta-model
kf1= KFold(n_splits=10, shuffle=True, random_state=42)
from sklearn.linear_model import LassoCV


# Create LassoCV model with a range of alpha values
alphas = np.logspace(-6, 6, 13)  # Adjust the range based on your specific needs
lasso_cv = LassoCV(alphas=alphas, cv=kf1)

# Fit the LassoCV model
lasso_cv.fit(X, y)

# Get the optimal alpha
optimal_alpha = lasso_cv.alpha_
print(f'Optimal Alpha: {optimal_alpha}')

# Fit the final Lasso model with the optimal alpha
meta_model= Lasso(alpha=optimal_alpha)
# Train the meta-model on stacked predictions
meta_model.fit(stacked_predictions, y)
# Prepare test data
test_data = df_imputed[columns] 
# Make predictions using base models
pred1_test = model1.predict(test_data)
pred2_test = model2.predict(test_data)
pred3_test = model3.predict(test_data)
pred4_test = model4.predict(test_data)
pred5_test = model5.predict(test_data)
pred6_test = model6.predict(test_data)
pred7_test = model7.predict(test_data)

pred9_test = model9.predict(test_data)
pred10_test = model10.predict(test_data)
pred11_test = model11.predict(test_data)
# Stack the predictions
stacked_predictions_test = np.column_stack((pred1_test, pred2_test, pred3_test, pred4_test,pred5_test,pred6_test,pred7_test,pred9_test,pred10_test,pred11_test))

# Use the meta-model to make final predictions
final_predictions_test = meta_model.predict(stacked_predictions_test)
y_test=df_imputed['PM2.5']
from sklearn.metrics import mean_squared_error, r2_score
final_rmse = mean_squared_error(y_test, final_predictions_test, squared=False)
mse_stacked = mean_squared_error(y_test, final_predictions_test)
rmse_stacked = np.sqrt(mse_stacked)
r2_stacked = r2_score(y_test, final_predictions_test)
print(f"Final RMSE on the test set: {final_rmse}")
print(f"Final RMSE_STACK on the test set: {rmse_stacked}")
print(f"Final MSE on the test set: {mse_stacked}")
print(f"Final r2 on the test set: {r2_stacked}")

plt.scatter(y_test,final_predictions_test)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.show()
