TELO CUSTOMER Churn

1. project overview:
    the telocustomer Churn prediction project aims to predict wheather a customer is likely to leave a telecommunication company or continue using its service.
    The main objective of this project is to develop a predictive model using customer information and deploy it through an interactive Streamlit web application.

2. Data description:
  The dataset used in this project is the telco customer churn dataset,obtained from kaggle.
source of data:https://www.kaggle.com/datasets/blastchar/telco-customer-churn


this dataset contain 7043 rows customer records and 21 columns.
Column name:
customerID:Unique customer ID
gender:Customer's gender
SeniorCitizen	:Whether customer is a senior citizen
Partner	:Whether customer has a partner
Dependents:	Whether customer has dependents
MultipleLines:whether the customer has multiple phone lines.
tenure:	Number of months as a customer
PhoneService:	Whether phone service is used
InternetService:	Type of internet service
Contract:	Customer's contract type
MonthlyCharges:Monthly bill amount
TotalCharges:	Total amount charged
Churn	: Whether customer left the company

3. Data Cleaning and Preprocessing:
Explain:
- Checked missing values.
- Converted TotalCharges into numeric format.
- Handled missing values.
- remove duplicate values.
- Removed unnecessary customerID column.

4. Exploratory Data Analysis (EDA)
   insights  discovered:
- Customer churn distribution.
- Tenure and churn relationship.
- Monthly charges and churn.
- Internet service and churn.
- Contract type and churn.
- Correlation between Monthly Charges and Total Charges through heatmap.
- gender distribution.

5. Feature Selection:
- Converts the target variable Churn from Yes/No into 1/0.
- One-Hot Encoding was applied using Pandas get_dummies() to convert categorical variables into numerical format.
- The dataset was then divided into training and testing sets using an 80:20 ratio. 
- Selects top 10 features for customer churn, and created a new dataset called X_selected for subsequent model development.

6. model devlopment:
- Feature Selection: Separate independent variables (X) and dependent variable (y).
- Data Encoding: Convert categorical variables into numerical format using pd.get_dummies().
- Train-Test Split: Divide the dataset into training (80%) and testing (20%) sets.
- Model Selection: Choose Random Forest Classifier.
-  Model Training: Train the model using model.fit(X_train, y_train).
-  Cross-Validation: Evaluate model consistency across different data splits.
-  Hyperparameter Tuning: Find suitable parameter values using GridSearchCV.
- After training, we evaluated its performance using Accuracy, Precision, Recall, and F1-score. We also generated a confusion matrix using Seaborn to visualize the correct and incorrect predictions.
- The model achieved 79.42% accuracy, 65.11% precision, 48.40% recall, and 55.52% F1-score. 


7. Model evaluation:
#evaluation matric:
- Accuracy = Overall correctness.
- Precision = Correctness of positive predictions.
- Recall = Completeness of positive predictions.
- F1 Score = Balance between Precision and Recall.

- The trained Random Forest model was used to predict customer churn on the test dataset.
- The predicted values (y_pred) were compared with the actual values (y_test).
- Accuracy was calculated to measure the percentage of correct predictions.
- Precision was used to measure how many predicted churn customers were actually churn customers.
- Recall was used to measure how many actual churn customers were correctly identified.
- F1-score was calculated to find the balance between precision and recall.
- A classification report was generated to display precision, recall, F1-score, and support for each class.
- GridSearchCV was used to find the best hyperparameter combination and improve model performance.
- The best model was selected based on its cross-validation F1-score.

8. app.py 
- Page Configuration: Sets the application title, page icon, and wide layout using Streamlit.
- Application Title: Displays the project title and a short introduction explaining the purpose of the application.
- Dataset Loading and Model Training: Loads the Telco dataset, removes customerID, converts TotalCharges into numeric format, handles missing values, encodes categorical data, and trains a Random Forest Classifier using a Pipeline. @st.cache_resource prevents unnecessary model retraining during app reruns.
- Customer Input Form: Provides an interactive form where users enter personal details, service information, and account details using Streamlit input widgets.
-  Prediction: Collects user inputs into a Pandas DataFrame and uses the trained model to predict whether the customer will churn or stay.
-  Display Prediction Result: Displays the prediction using success or error messages, along with a short explanation.
-  Prediction Probability: Displays the estimated probability of churn and no churn as percentages with progress bars.
- Footer: Displays a divider and project information at the bottom of the application.

9. Tools and Libraries Used:
Streamlit: To create the interactive web application.
Pandas: To organize user input into a DataFrame.
Joblib: To load the saved machine learning model.
Scikit-learn: To perform machine learning predictions through the trained model.

* Deployment Result:
The trained Random Forest model was successfully deployed using Streamlit. Users can enter customer information and receive predictions about whether the customer is likely to leave the company (Churn: Yes) or stay with the company (Churn: No).
