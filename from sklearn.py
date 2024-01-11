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
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
            
params_gb={
       
            'n_estimators': 800,
            'max_depth': 5,
            'learning_rate': 0.01}
model9 = GradientBoostingRegressor(**params_gb)
model9.fit(X_train, y_train)
y_pred=model9.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
print(mse)
print(r2)