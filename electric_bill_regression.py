import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────
# Step 1: Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('electric_bill.csv')
print('✅ Dataset loaded!')
print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')
print(df.head())

# ─────────────────────────────────────────
# Step 2: Explore Dataset
# ─────────────────────────────────────────
print('\n📊 Statistical Summary:')
print(df.describe())

print('\n🔍 Data Types:')
print(df.dtypes)

# ─────────────────────────────────────────
# Step 3: Handle Missing Values
# ─────────────────────────────────────────
print('\n🔍 Null Values:')
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
print('✅ Missing values handled!')

# ─────────────────────────────────────────
# Step 4: Select Features and Target
# ─────────────────────────────────────────
X = df.drop('Monthly_Bill_PHP', axis=1)
y = df['Monthly_Bill_PHP']

print('\nFeatures (X) — first 5 rows:')
print(X.head())
print('\nTarget (y) — first 5 rows:')
print(y.head())

# ─────────────────────────────────────────
# Step 5: Split Dataset
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\n✅ Train size: {X_train.shape[0]} rows (80%)')
print(f'✅ Test size:  {X_test.shape[0]} rows (20%)')

# ─────────────────────────────────────────
# Step 6: Build Model
# ─────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('\n✅ Model trained successfully!')

# ─────────────────────────────────────────
# Step 7: Evaluate Model
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print('\n📊 Regression Evaluation Metrics')
print('='*40)
print(f'MAE  (Mean Absolute Error):       ₱{mae:,.2f}')
print(f'MSE  (Mean Squared Error):        ₱{mse:,.2f}')
print(f'RMSE (Root Mean Squared Error):   ₱{rmse:,.2f}')
print(f'R²   (Coefficient of Det.):       {r2:.4f}')

# ─────────────────────────────────────────
# Step 8: Visualize
# ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='steelblue', edgecolors='white', s=100)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Bill (₱)')
plt.ylabel('Predicted Bill (₱)')
plt.title('Actual vs Predicted Monthly Electric Bill')
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='steelblue')
plt.title('Feature Importances')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# Step 9: Save Model
# ─────────────────────────────────────────
joblib.dump(model, 'electric_bill_model.pkl')
print('\n✅ Model saved as electric_bill_model.pkl')
