# Simple Machine Learning Program: Predict House Prices
from sklearn.linear_model import LinearRegression

# Step 1: Training Data (house size vs price)
# X -> feature (size of house in sq. ft.)
# y -> target (price in lakhs)
X = [[500], [1000], [1500], [2000], [2500]]
y = [50, 100, 150, 200, 250]

# Step 2: Create and Train the Model
model = LinearRegression()
model.fit(X, y)

# Step 3: Make Predictions
predicted_price = model.predict([[3000]])  # Predict price for 3000 sq. ft house

print("Predicted price for 3000 sq. ft house:", predicted_price[0], "lakhs")
