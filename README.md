#  Student Performance Predictor

A supervised Machine Learning project that predicts whether a student will **Pass or Fail** based on demographic and behavioral features.

##  Features Used
| Feature | Description |
|---|---|
| gender | Male / Female |
| age | Student age |
| study_hours | Daily study hours |
| attendance | Attendance percentage |
| parental_edu | Parent's education level |
| internet_access | Internet availability |
| extra_activities | Participation in activities |
| prev_score | Previous exam score |

##  Models Trained
- Logistic Regression
- Decision Tree Classifier
- **Random Forest Classifier ← Best (84% accuracy)**

## Outputs
- Model accuracy comparison chart
- Confusion matrix
- Feature importance plot
- ROC curve for all models
- Prediction for a new student

##  How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the script
python student_performance.py
```

Or open `student_performance.ipynb` in **Google Colab / Jupyter Notebook**.

##  Project Structure
```
student_performance_predictor/
├── student_performance.py     
├── student_performance.ipynb   
├── requirements.txt
└── README.md
```

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`
