import streamlit as st 
import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor 
import pickle



#Loads the Boston House Price Dataset
boston = fetch_california_housing()



X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target,columns = boston.target_names)


#Build model
model = RandomForestRegressor()
model.fit(X,Y)

#saving model
pickle.dump(model,open('calif_house.pkl','wb'))











    
    
