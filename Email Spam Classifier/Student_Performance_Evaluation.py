# import statements
import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# -- DOWNLOAD DATA
print("Downloading dataset...")
spscientist_students_performance_in_exams_path = kagglehub.dataset_download('spscientist/students-performance-in-exams')
print('Data source import complete.\n')

# 3. PARAMETERS
passmark = 40

# 4. LOAD DATA
print("Loading dataset...")
df = pd.read_csv(f"{spscientist_students_performance_in_exams_path}/StudentsPerformance.csv")
print("Dataset loaded successfully.\n")

# 5. EDA & BASIC INFO
print("---- Basic EDA ----")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nStatistical summary of numerical columns:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())

# 6. EXPLORATORY PLOTS
print("---- Plotting Score Distributions ----")
sns.countplot(x="math score", data=df, palette="muted")
plt.xticks(rotation=90)
plt.title("Distribution of Math Scores")
plt.show()

sns.countplot(x="reading score", data=df, palette="muted")
plt.xticks(rotation=90)
plt.title("Distribution of Reading Scores")
plt.show()

sns.countplot(x="writing score", data=df, palette="muted")
plt.xticks(rotation=90)
plt.title("Distribution of Writing Scores")
plt.show()

# 7. PASS/FAIL STATUS FOR EACH SUBJECT
print("---- Adding Pass/Fail Status for Each Subject ----")
df['Math_PassStatus'] = np.where(df['math score'] < passmark, 'F', 'P')
df['Reading_PassStatus'] = np.where(df['reading score'] < passmark, 'F', 'P')
df['Writing_PassStatus'] = np.where(df['writing score'] < passmark, 'F', 'P')

print("\nPass/Fail status counts for Math:")
print(df.Math_PassStatus.value_counts())
print("\nPass/Fail status counts for Reading:")
print(df.Reading_PassStatus.value_counts())
print("\nPass/Fail status counts for Writing:")
print(df.Writing_PassStatus.value_counts())

# 8. PASS STATUS VISUALIZATIONS
print("---- Plotting Pass/Fail Status by Parental Level of Education ----")
sns.countplot(x='parental level of education', data=df, hue='Math_PassStatus', palette='bright')
plt.xticks(rotation=90)
plt.title("Math Pass Status by Parental Level of Education")
plt.show()

sns.countplot(x='parental level of education', data=df, hue='Reading_PassStatus', palette='bright')
plt.xticks(rotation=90)
plt.title("Reading Pass Status by Parental Level of Education")
plt.show()

sns.countplot(x='parental level of education', data=df, hue='Writing_PassStatus', palette='bright')
plt.xticks(rotation=90)
plt.title("Writing Pass Status by Parental Level of Education")
plt.show()

# 9. OVERALL PASS STATUS
print("---- Adding Overall Pass/Fail Status ----")
df['OverAll_PassStatus'] = df.apply(
    lambda x: 'F' if (x['Math_PassStatus'] == 'F' or x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F') else 'P',
    axis=1
)
print("\nOverall Pass/Fail status counts:")
print(df.OverAll_PassStatus.value_counts())

sns.countplot(x='parental level of education', data=df, hue='OverAll_PassStatus', palette='bright')
plt.xticks(rotation=90)
plt.title("Overall Pass Status by Parental Level of Education")
plt.show()

# 10. PERCENTAGE, GRADES, AND VISUALIZATION
print("---- Calculating Percentage and Grades ----")
df['Total_Marks'] = df['math score'] + df['reading score'] + df['writing score']
df['Percentage'] = df['Total_Marks'] / 3

# Grade function: above 80=A, 70-80=B, 60-70=C, 50-60=D, 40-50=E, below 40=F
def GetGrade(Percentage, OverAll_PassStatus):
    if OverAll_PassStatus == 'F':
        return 'F'
    if Percentage >= 80:
        return 'A'
    if Percentage >= 70:
        return 'B'
    if Percentage >= 60:
        return 'C'
    if Percentage >= 50:
        return 'D'
    if Percentage >= 40:
        return 'E'
    return 'F'

df['Grade'] = df.apply(lambda x: GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)
print("\nGrade distribution:")
print(df.Grade.value_counts())

sns.countplot(x="Grade", data=df, order=['A','B','C','D','E','F'], palette="muted")
plt.title("Grade Distribution")
plt.show()

sns.countplot(x='parental level of education', data=df, hue='Grade', palette='bright')
plt.xticks(rotation=90)
plt.title("Grade by Parental Level of Education")
plt.show()

# 11. SIMPLE ML MODELING: KNN & LOGISTIC REGRESSION
print("---- Preparing Data for Classification Models (KNN & Logistic Regression) ----")
# For quick demo, use pass/fail overall as target; encode as binary
df['target_class'] = (df['OverAll_PassStatus'] == 'P').astype(int)
features = ['math score', 'reading score', 'writing score']
X = df[features]
y_class = df['target_class']

# Train/test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=0)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classification
print("Training KNN Classification model...")
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train_class)
y_pred_knn = knn.predict(X_test_scaled)
print("\nKNN Classification Report:")
print(classification_report(y_test_class, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test_class, y_pred_knn))

# Logistic Regression Classification
print("Training Logistic Regression model...")
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train_class)
y_pred_log = log_reg.predict(X_test_scaled)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_class, y_pred_log))
print("Logistic Regression Accuracy:", accuracy_score(y_test_class, y_pred_log))

# 12. Regression Example: Predict Percentage
print("---- Training Regression Model (Predict Percentage) ----")
from sklearn.linear_model import LinearRegression
y_reg = df['Percentage']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=0)

print("Training Linear Regression model...")
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)
print("\nLinear Regression Mean Squared Error (MSE):", mean_squared_error(y_test_reg, y_pred_reg))
