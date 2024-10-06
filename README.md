# Home-Loan-Approval-Prediction
### Data Source
- **Home Loan Approval Dataset** ([Kaggle link](https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval))

### Goal
Build a machine learning classification model to predict whether a loan will be approved based on various features such as applicant income, credit history, and more.

### Summary
- Collected home loan data from Kaggle.
- Preprocessed the data by:
  - Handling missing values.
  - One-hot encoding categorical variables.
  - Standardizing features.
- Built and evaluated multiple classification models:
  - **K-Nearest Neighbors (KNN)**
  - **Naive Bayes (GNB)**
  - **Support Vector Machines (Linear, RBF, Poly, Sigmoid)**
  - **Decision Tree (DT)**
  - **Random Forest (RF)**
- Performed 5-fold cross-validation for model validation.
- Generated data visualizations:
  - Correlation matrix.
  - Joint plots and density plots of applicant income vs. loan amount and loan status.

### Results
- **Best Performing Model**: Linear SVM
  - Accuracy: **83%**
  - Cross-validation score: **85%**
- Other notable models: Gaussian Naive Bayes with similar accuracy.
- Insights: Applicant income and credit history showed the strongest correlation with loan approval.

### Tech Stack
- **Language**: Python
- **Libraries**: Pandas, Matplotlib, Seaborn, Scikit-learn
- **Tool**: Google Colab
