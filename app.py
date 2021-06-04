
# Load the dataset.
# Import the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load the dataset.
file_path = 'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/glass-types.csv'
df = pd.read_csv(file_path, header = None)

# # Drop the 0th column as it contains only the serial numbers.
df.drop(columns = 0, inplace = True)

# A Python list containing the suitable column headers as string values. Also, create a Python dictionary as described above.
column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']

# Required Python dictionary.
columns_dict = {}
for i in df.columns:
  columns_dict[i] = column_headers[i - 1]

# Rename the columns.
df.rename(columns_dict, axis = 1, inplace = True)

# Create separate data-frames for training and testing the model.
from sklearn.model_selection import train_test_split

# Creating the features data-frame holding all the columns except the last column
x = df.iloc[:, :-1]


# Creating the target series that holds last column 'RainTomorrow'
y = df['GlassType']


# Splitting the train and test sets using the 'train_test_split()' function.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Build a logistic regression model using the 'sklearn' module.
from sklearn.linear_model import LogisticRegression

# 1. First, call the 'LogisticRegression' module and store it in lg_clg
lg_clf = LogisticRegression()

# 2. Call the 'fit()' function with 'x_train' and 'y_train' as inputs.
lg_clf.fit(x_train, y_train)

# 3. Call the 'score()' function with 'x_train' and 'y_train' as inputs to check the accuracy score of the model.
score=lg_clf.score(x_train, y_train)


import streamlit as st


# defining the function which will make the prediction using the data which the user inputs 
def prediction(ri, na, mg, al, si, k, ca, ba, fe):   
 

 
    # Making predictions 
    pr = lg_clf.predict( 
        [[ri, na, mg, al, si, k, ca, ba, fe]])
     
    return pr
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Glass Type prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp,unsafe_allow_html=True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    
    Ri = st.slider("Input Ri",0.0,2.0)
    Na = st.slider("Input Na",10.0,20.0)
    Mg = st.slider("Input Mg",0.0,10.0)
    Al = st.slider("Input Al",0.0,5.0)
    Si = st.slider("Input Si",60.0,80.0)
    K = st.slider("Input K",0.0,2.0)
    Ca = st.slider("Input Ca",0.0,10.00)
    Ba = st.slider("Input Ba",0.0,20.0)
    Fe = st.slider("Input Fe",0.0,2.0)

    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"):
        glass_type = prediction(Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        
        st.write('type of glass is {}'.format(glass_type))
        
        st.write('Accuracy score is {}'.format(score))
     
if __name__=='__main__': 
    main()
