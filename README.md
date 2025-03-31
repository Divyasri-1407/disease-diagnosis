import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample dataset (Replace with real medical dataset for better accuracy)
data = {
    'fever': [1, 0, 1, 0, 1, 0, 1, 1],
    'cough': [1, 1, 0, 0, 1, 1, 1, 0],
    'fatigue': [0, 1, 1, 0, 1, 0, 1, 1],
    'headache': [1, 0, 1, 1, 0, 0, 1, 1],
    'disease': ['Flu', 'Cold', 'Flu', 'Migraine', 'Flu', 'Cold', 'Flu', 'Migraine']
}

df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
df['disease'] = label_encoder.fit_transform(df['disease'])

# Split data into features and target variable
X = df.drop(columns=['disease'])
y = df['disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Function to make predictions
def predict_disease(symptoms):
    input_data = pd.DataFrame([symptoms], columns=X.columns)
    prediction = model.predict(input_data)
    disease_name = label_encoder.inverse_transform(prediction)[0]
    return disease_name

# Example usage
symptoms = {'fever': 1, 'cough': 1, 'fatigue': 0, 'headache': 1}
predicted_disease = predict_disease(symptoms)
print(f'Predicted Disease: {predicted_disease}')
