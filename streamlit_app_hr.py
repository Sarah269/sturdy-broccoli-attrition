
# Load Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="IBM HR Attrition ML App",
    page_icon=":runner:",
    layout="wide",
    initial_sidebar_state="expanded")

# Write Title for Streamlit App
st.write("""
### HR Employee Attrition Machine Learning App 

This app predicts the probability of an employee leaving the company 

* **Data Source**: Kaggle
* **Dataset**: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Dataset Details**: 46 features (Used 17). 1,470 rows. Imbalanced
* **Classification Model**:  RandomForestClassifier
* **Model Preparation**:  Feature Engineering. Data Resampling. Hyperparameter Tuning
 

          
""")


# Metrics
st.write("Attrition Metrics")

col4,col5, col6, col7 =  st.columns((4,4,4,4))

with col4:
   tile = st.container(height=120)
   tile.metric(label=" :runner: Corporate", value = "13.9%") 

with col5:
   tile = st.container(height=120)
   tile.metric(label=" :runner: Men", value = "17%") 

with col6:
   tile = st.container(height=120)
   tile.metric(label=" :runner: Women", value = "14.8%") 

with col7:
   tile = st.container(height=120)
   tile.metric(label=" :runner: Single", value = "25.5%") 

col1, col2, col3 = st.columns((5,5,5))
with col1:
    # Human Resources Attrition
    st.image("HRplot.png")

with col2:
    # Research & Development Attrition
    st.image("RDplot.png")

with col3:
    # Sales Attrition
    st.image("Salesplot.png")


st.sidebar.write("**User Input Parameters**")

# Define User Input Feature Function
def user_input_features():

   st.sidebar.write("**General** ")
   
   Age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=36)
   MaritalStatus = st.sidebar.selectbox("Marital Status",("Single","Married","Other"))
   if MaritalStatus == "Single":
      MS_Single = 1
      MS_Married = 0
   elif MaritalStatus == "Married":
      MS_Single = 0
      MS_Married = 1
   else:
      MS_Single = 0
      MS_Married = 0

   TotalWorkingYears = st.sidebar.number_input("Total Working Years",min_value=0,value=1)

   st.sidebar.write("**Compensation**")
   
   MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=0, value=6500)
   WorksOT = st.sidebar.selectbox("Works OverTime",("No","Yes"))
   if WorksOT == "No":
      OverTime = 0
   else:
      OverTime = 1
   StockOptionLevel = st.sidebar.selectbox("Stock Option Level",(0,1,2,3))

   st.sidebar.write("Company")
   
   JobLevel = st.sidebar.selectbox("Job Level",(1,2,3,4,5))
   JobSatisfaction = st.sidebar.selectbox("Job Satisfaction",(1,2,3,4))
   YearsAtCompany = st.sidebar.number_input("Years at company",min_value=0,value=1)
   YearsInCurrentRole = st.sidebar.number_input("Years in Current Role",min_value=0,value=1)
   YearsWithCurrManager = st.sidebar.number_input("Years With Current Manager",min_value=0,value=2)
   Travel= st.sidebar.selectbox("Travel",("No Travel","Frequently", "Other"))
   match Travel:
      case "No Travel":
         BT_NonTravel = 1
         BT_Travel_Frequently = 0
      case "Frequently":
         BT_NonTravel = 0
         BT_Travel_Frequently = 1
      case "Other":
         BT_NonTravel = 0
         BT_Travel_Frequently = 0
      case _:
         BT_NonTravel = 0
         BT_Travel_Frequently = 0

   JobRole = st.sidebar.selectbox("Job Role",("Laboratory Technician","Manager", "Sales Representative", "Other"))
   match JobRole:
      case "Laboratory Technician":
         JR_LaboratoryTechnician = 1
         JR_Manager = 0
         JR_SalesRepresentative = 0
      case "Manager":
         JR_LaboratoryTechnician = 0
         JR_Manager = 1
         JR_SalesRepresentative = 0
      case "Sales Representative":
         JR_LaboratoryTechnician = 0
         JR_Manager = 0
         JR_SalesRepresentative = 1
      case "Other":
         JR_LaboratoryTechnician = 0
         JR_Manager = 0
         JR_SalesRepresentative = 0
      case _:
         JR_LaboratoryTechnician = 0
         JR_Manager = 0
         JR_SalesRepresentative = 0
   

   data = {
      'Age': Age,
      'JobLevel': JobLevel,
      'JobSatisfaction': JobSatisfaction,
      'MonthlyIncome': MonthlyIncome,
      'OverTime': OverTime,
      'StockOptionLevel': StockOptionLevel,
      'TotalWorkingYears': TotalWorkingYears,
      'YearsAtCompany': YearsAtCompany,
      'YearsInCurrentRole': YearsInCurrentRole,
      'YearsWithCurrManager': YearsWithCurrManager,
      'BT_NonTravel': BT_NonTravel,
      'BT_Travel_Frequently': BT_Travel_Frequently,
      'JR_LaboratoryTechnician': JR_LaboratoryTechnician ,
      'JR_Manager':  JR_Manager,
      'JR_SalesRepresentative': JR_SalesRepresentative ,
      'MS_Married': MS_Married ,
      'MS_Single': MS_Single
      
   }

   features = pd.DataFrame(data, index=[0])
   return features

# Capture user selected features from sidebar
df_input = user_input_features()

st.write("___")  
st.write(""" 
         
         **User Parameters for HR Attrition**
         
          """)

st.dataframe(df_input)
st.write("___")  

# Prediction
if st.sidebar.button("Predict"):
    # Load Model
    # Cache resource so that is loads once
    @st.cache_resource
    def load_model():
        model = joblib.load('HRattrition_model_jl.sav.bz2')
        return model

    load_joblib_model = load_model()

    # Predict with model
    prediction = load_joblib_model.predict(df_input)
    predict_proba = load_joblib_model.predict_proba(df_input)

    # List Prediction outcome
    st.subheader('Prediction ')
    # st.write(prediction)
    if prediction == 0:
        st.write("Employee will stay")
    elif prediction == 1:
        st.write("Employee will leave")
    else:
        st.write("Oops...unexpected result...Try again")

    # List Prediction Probability

    st.subheader('Prediction Probability')
  
    df_prediction_proba = pd.DataFrame(predict_proba)
    df_prediction_proba.columns = ['No Attrition', 'Attrition']
    df_prediction_proba.rename(columns={0: 'No Attrition',
                                    1: 'Attrition'})
                                
    # Progress Bar for Attrition Probability
    st.dataframe(df_prediction_proba,
                column_config={
                'No Attrition': st.column_config.ProgressColumn(
                    'No Attrition',
                    format='%.2f',
                    width='medium',
                    min_value=0,
                    max_value=1
                ),
                'Attrition': st.column_config.ProgressColumn(
                    'Attrition',
                    format='%.2f',
                    width='medium',
                    min_value=0,
                    max_value=1
                ),
                
                }, hide_index=True)