import pandas as pd
from sklearn.ensemble import IsolationForest
import boto3


# Load dataset from S3
s3 = boto3.client('s3')
bucket = 'a208321-ml-workspace-6fede99ec6de-use1'
key = '2024/05/12/training_dataset_final.csv'
local_file = 'training_dataset_final.csv'
s3.download_file(bucket, key, local_file)

# Read the dataset
df = pd.read_csv(local_file)

# Assume you're interested in examining anomalies in the following features
features = df[['TAX_YEAR','REFUND','STATE_PLACEHOLDER', 'STATE_ATTACHED', 'ZIPCODE']]  # Update with your actual feature columns

# Setting up the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)  # contamination is an estimate of the anomaly proportion in the data

# Fitting the model
iso_forest.fit(features)

# Predicting anomalies (-1 labels anomalies)
df['anomaly'] = iso_forest.predict(features)

# Filter out the anomalies (potential false positives)
anomalies = df[df['anomaly'] == -1]

# Display anomalies
print(anomalies)

# Optionally, save the anomalies to a new CSV file
anomalies.to_csv('anomalies.csv', index=False)