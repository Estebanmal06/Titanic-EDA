from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

# Initialize the API
api = KaggleApi()
api.authenticate()

# Download Titanic dataset directly from Kaggle
api.competition_download_files('titanic', path='./data')

# Manually unzip the downloaded dataset
zip_file_path = './data/titanic.zip'
unzip_dir = './data/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# Now, you can load the Titanic dataset
import pandas as pd

# Load the Titanic dataset
train_df = pd.read_csv('./data/train.csv')

# Display the first few rows of the dataset
print(train_df.head())

# Check for missing values
print(train_df.isnull().sum())

# Display summary statistics
print(train_df.describe())

# Analyze survival distribution
print(train_df['Survived'].value_counts())

# Visualize the data
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=train_df)
plt.title('Survival Distribution')
plt.show()

# Example: Gender vs Survival
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.title('Survival by Gender')
plt.show()


