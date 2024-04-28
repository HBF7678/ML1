import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Define the dataset
data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook': ['SUNNY', 'SUNNY', 'OVERCAST', 'RAIN', 'RAIN', 'RAIN', 'OVERCAST', 'SUNNY', 'OVERCAST', 'RAIN', 'SUNNY', 'OVERCAST', 'OVERCAST', 'RAIN'],
    'Temperature': ['HOT', 'HOT', 'HOT', 'MILD', 'COOL', 'COOL', 'COOL', 'HOT', 'COOL', 'MILD', 'MILD', 'HOT', 'HOT', 'MILD'],
    'Humidity': ['HIGH', 'HIGH', 'HIGH', 'HIGH', 'NORMAL', 'NORMAL', 'NORMAL', None, None, 'NORMAL', 'HIGH', 'HIGH', 'NORMAL', 'HIGH'],
    'Wind': ['WEAK', 'STRONG', 'WEAK', 'WEAK', 'WEAK', 'STRONG', 'STRONG', 'STRONG', 'STRONG', 'WEAK', 'STRONG', 'STRONG', 'WEAK', 'STRONG'],
    'Play Ball': ['NO', 'NO', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'NO']
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Fill missing values in 'Humidity' column with the most frequent value
df['Humidity'].fillna(df['Humidity'].mode()[0], inplace=True)

# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
df['Outlook'] = label_encoder.fit_transform(df['Outlook'])
df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
df['Humidity'] = label_encoder.fit_transform(df['Humidity'])
df['Wind'] = label_encoder.fit_transform(df['Wind'])

# Create a OneHotEncoder to apply to 'Outlook', 'Temperature', 'Humidity', 'Wind'
onehot_encoder = ColumnTransformer([('encoder', OneHotEncoder(), ['Outlook', 'Temperature', 'Humidity', 'Wind'])], remainder='passthrough')

# Apply the one-hot encoding
df_encoded = onehot_encoder.fit_transform(df)

# Split the data into training and testing sets
X = df_encoded[:, :-1]  # Features
y = df_encoded[:, -1]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Return the shapes of training and testing sets
