# Create a DataFrame
import pandas as pd
df = pd.DataFrame({
    'Day': ['D1', 'D2', ...],  # Day identifiers
    'Outlook': ['SUNNY', 'SUNNY', ...],  # Categorical attributes
    'Temperature': ['HOT', 'HOT', ...],
    'Humidity': ['HIGH', 'HIGH', ...],
    'Wind': ['WEAK', 'STRONG', ...],
    'Play Ball': ['NO', 'NO', 'YES', ...]  # Target variable
})

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Ball']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Create and fit the Naive Bayes model
from sklearn.naive_bayes import CategoricalNB
nb_model = CategoricalNB()
nb_model.fit(df[['Outlook', 'Temperature', 'Humidity', 'Wind']], df['Play Ball'])

# Define the unknown sample
unknown_sample = {
    'Outlook': 'OVERCAST',
    'Temperature': 'MILD',
    'Humidity': 'NORMAL',
    'Wind': 'STRONG'
}

# Encode the unknown sample and predict
encoded_sample = {key: label_encoders[key].transform([value])[0] for key, value in unknown_sample.items()}
prediction = nb_model.predict([list(encoded_sample.values())])
