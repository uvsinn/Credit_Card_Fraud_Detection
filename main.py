import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
credit_card_data=pd.read_csv('credit_card_data.csv')

#DATA PREPROCESSING
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
legit_sample=legit.sample(n=492)

#SPLITTING THE DATA
new_dataset=pd.concat([legit_sample,fraud],axis=0)
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=40)

#MODEL
model=LogisticRegression()
model.fit(X_train,Y_train)

#EVALUATION
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)

#WEBAPP
st.title("Credit Card Fraud Detection")
input_df=st.text_input('Enter all required features values')
input_df_splitted= input_df.split(',')

submit=st.button("Submit")

if submit:
    features=np.asarray(input_df_splitted, dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0]==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")

st.subheader('Accuracy: ')
st.write(str(accuracy_score(Y_test, model.predict(X_test))*100)+'%')
