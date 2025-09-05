import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set color palette to match the website theme
plt.style.use('default')
primary_color = '#2193b0'
secondary_color = '#6dd5ed'
accent_colors = ['#2193b0', '#6dd5ed', '#176a80', '#4db8d8']

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Read the data
df = pd.read_csv('forms/anemia.csv')
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

# Figure 1: Original dataset distribution
plt.figure(figsize=(10, 6))
results = df['Result'].value_counts()
bars = plt.bar(['No Anemia (0)', 'Anemia (1)'], results.values, 
               color=[primary_color, secondary_color], 
               edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, value in zip(bars, results.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.xlabel('Anemia Status', fontsize=12, fontweight='bold')
plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
plt.title('Original Dataset Distribution\n(Before Balancing)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/Figure_1.png', dpi=300, bbox_inches='tight')
plt.close()

# Data balancing
from sklearn.utils import resample

# Separate majority and minority classes
majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]

# Undersample the majority class to match the minority class size
major_downsample = resample(
    majorclass,
    replace=False,
    n_samples=len(minorclass),
    random_state=42
)

# Combine downsampled majority class with minority class
df = pd.concat([major_downsample, minorclass])

# Figure 2: Balanced dataset distribution
plt.figure(figsize=(10, 6))
results_balanced = df['Result'].value_counts()
bars = plt.bar(['No Anemia (0)', 'Anemia (1)'], results_balanced.values,
               color=[primary_color, secondary_color], 
               edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, value in zip(bars, results_balanced.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.xlabel('Anemia Status', fontsize=12, fontweight='bold')
plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
plt.title('Balanced Dataset Distribution\n(After Undersampling)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/Figure_2.png', dpi=300, bbox_inches='tight')
plt.close()

print(df['Result'].value_counts())
print(df.describe())

# Figure 3: Gender distribution
plt.figure(figsize=(10, 6))
gender_counts = df['Gender'].value_counts()
bars = plt.bar(['Male (0)', 'Female (1)'], gender_counts.values,
               color=[accent_colors[2], accent_colors[3]], 
               edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, value in zip(bars, gender_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.xlabel('Gender', fontsize=12, fontweight='bold')
plt.ylabel('Number of Participants', fontsize=12, fontweight='bold')
plt.title('Gender Distribution in Dataset', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/Figure_3.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Hemoglobin distribution
plt.figure(figsize=(12, 6))
plt.hist(df['Hemoglobin'], bins=25, color=primary_color, alpha=0.7, 
         edgecolor='white', linewidth=1)
plt.axvline(df['Hemoglobin'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {df["Hemoglobin"].mean():.1f}')
plt.xlabel('Hemoglobin Level (g/dL)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Hemoglobin Levels', fontsize=14, fontweight='bold', pad=20)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/Figure_4.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Mean Hemoglobin by Gender and Result
plt.figure(figsize=(12, 8))
grouped_data = df.groupby(['Gender', 'Result'])['Hemoglobin'].mean().unstack()

# Create positions for bars
x = np.arange(2)
width = 0.35

bars1 = plt.bar(x - width/2, grouped_data[0], width, 
                label='No Anemia', color=primary_color, alpha=0.8)
bars2 = plt.bar(x + width/2, grouped_data[1], width,
                label='Anemia', color=secondary_color, alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Gender', fontsize=12, fontweight='bold')
plt.ylabel('Average Hemoglobin Level (g/dL)', fontsize=12, fontweight='bold')
plt.title('Average Hemoglobin Levels by Gender and Anemia Status', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, ['Male', 'Female'])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/Figure_5.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Pairplot with custom styling
plt.figure(figsize=(15, 12))
# Create a custom pairplot
feature_cols = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
pairplot = sns.pairplot(df[feature_cols + ['Result']], hue='Result', 
                       palette=[primary_color, secondary_color],
                       plot_kws={'alpha': 0.6, 's': 40})
pairplot.fig.suptitle('Relationships Between Blood Parameters', 
                     fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('static/Figure_6.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Result']].corr()

# Create custom colormap
colors = ['#ff4757', '#ffffff', primary_color]
n_bins = 100
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
heatmap = sns.heatmap(correlation_matrix, 
                     annot=True, 
                     cmap=cmap,
                     center=0,
                     square=True,
                     mask=mask,
                     cbar_kws={"shrink": 0.8},
                     annot_kws={'fontsize': 10, 'fontweight': 'bold'})

plt.title('Correlation Matrix of Blood Parameters', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('static/Figure_7.png', dpi=300, bbox_inches='tight')
plt.close()

# Continue with the machine learning part
X = df.drop('Result', axis=1)
print(X)

Y = df['Result']
print(Y)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Print the shapes of the resulting datasets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred)
c_lr = classification_report(y_test, y_pred)

print('Logistic Regression Accuracy Score: ', acc_lr)
print(c_lr)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test, y_pred)
c_rf = classification_report(y_test, y_pred)

print('Random Forest Accuracy Score: ', acc_rf)
print(c_rf)

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)

acc_dt = accuracy_score(y_test, y_pred)
c_dt = classification_report(y_test, y_pred)

print('Decision Tree Accuracy Score: ', acc_dt)
print(c_dt)

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test, y_pred)
c_nb = classification_report(y_test, y_pred)

print('Naive Bayes Accuracy Score: ', acc_nb)
print(c_nb)

from sklearn.svm import SVC

support_vector = SVC()
support_vector.fit(x_train, y_train)
y_pred = support_vector.predict(x_test)

acc_svc = accuracy_score(y_test, y_pred)
c_svc = classification_report(y_test, y_pred)

print('SVM Accuracy Score: ', acc_svc)
print(c_svc)

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_test)

acc_gbc = accuracy_score(y_test, y_pred)
c_gbc = classification_report(y_test, y_pred)

print('Gradient Boosting Accuracy Score: ', acc_gbc)
print(c_gbc)

# Test prediction
prediction = GBC.predict([[0, 11.6, 22.3, 30.9, 74.5]])
print(f"Test prediction result: {prediction[0]}")
if prediction[0] == 0:
    print("The person is not affected by anemia")
elif prediction[0] == 1:
    print("The person is affected by anemia")  

# Model comparison
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier',
              'Gaussian Naive Bayes', 'Support Vector Classifier', 'Gradient Boost Classifier'],
    'Score': [acc_lr, acc_dt, acc_rf, acc_nb, acc_svc, acc_gbc],
})
print(model_comparison)

# Save the best model
import pickle
import warnings
pickle.dump(GBC, open('model.pkl', 'wb'))
warnings.filterwarnings("ignore")
print("Model saved successfully!")
print("All graphs have been saved to the 'static' folder!")