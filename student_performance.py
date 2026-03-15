import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ── 1. GENERATE SAMPLE DATASET 
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'gender':           np.random.choice(['Male', 'Female'], n),
    'age':              np.random.randint(15, 20, n),
    'study_hours':      np.round(np.random.uniform(1, 10, n), 1),
    'attendance':       np.random.randint(50, 100, n),
    'parental_edu':     np.random.choice(['None', 'High School', 'Graduate', 'Post-Graduate'], n),
    'internet_access':  np.random.choice(['Yes', 'No'], n),
    'extra_activities': np.random.choice(['Yes', 'No'], n),
    'prev_score':       np.random.randint(40, 100, n),
})

# Target: Pass (1) / Fail (0) — based on study hours + attendance + prev score
data['pass_fail'] = (
    (data['study_hours'] >= 4) &
    (data['attendance'] >= 70) &
    (data['prev_score'] >= 50)
).astype(int)

print("Dataset shape:", data.shape)
print("\nClass distribution:\n", data['pass_fail'].value_counts())
print("\nSample data:\n", data.head())

# ── 2. PREPROCESSING 
le = LabelEncoder()
cat_cols = ['gender', 'parental_edu', 'internet_access', 'extra_activities']
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('pass_fail', axis=1)
y = data['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\nTraining samples:", X_train.shape[0])
print("Test samples    :", X_test.shape[0])

# ── 3. TRAIN MODELS 
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    acc   = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}
    print(f"\n{name} Accuracy: {acc*100:.2f}%")

best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]
print(f"\nBest Model: {best_name} ({best['accuracy']*100:.2f}%)")

# ── 4. CLASSIFICATION REPORT 
print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, best['preds'],
                             target_names=['Fail', 'Pass']))

# ── 5. VISUALIZATIONS 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Performance Predictor — Model Analysis', fontsize=15, fontweight='bold')

# (a) Model Accuracy Comparison
names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]
axes[0, 0].bar(names, accs, color=['#555555', '#888888', '#222222'])
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_ylim(0, 110)
for i, v in enumerate(accs):
    axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# (b) Confusion Matrix
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'], ax=axes[0, 1])
axes[0, 1].set_title(f'Confusion Matrix — {best_name}')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# (c) Feature Importance (Random Forest)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', ax=axes[1, 0], color='#444444')
axes[1, 0].set_title('Feature Importance — Random Forest')
axes[1, 0].set_xlabel('Importance Score')

# (d) ROC Curve
for name, res in results.items():
    probs = res['model'].predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=150)
plt.show()
print("\nPlot saved as model_analysis.png")

# ── 6. PREDICT ON NEW STUDENT 
print("\n--- Predict for a New Student ---")
new_student = pd.DataFrame([{
    'gender': 1,           # Female
    'age': 17,
    'study_hours': 6.5,
    'attendance': 85,
    'parental_edu': 2,     # Graduate
    'internet_access': 1,  # Yes
    'extra_activities': 1, # Yes
    'prev_score': 72,
}])
new_scaled = scaler.transform(new_student)
prediction = rf_model.predict(new_scaled)[0]
probability = rf_model.predict_proba(new_scaled)[0][1]
print(f"Prediction  : {'PASS ✓' if prediction == 1 else 'FAIL ✗'}")
print(f"Pass Probability: {probability*100:.1f}%")
