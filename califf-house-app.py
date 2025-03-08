import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pickle
import seaborn as sns 



st.write("""
         # California House Price Prediction App
         
         This app predicts the **California House Price**
         
         """)



#Loads the Boston House Price Dataset
boston = fetch_california_housing()


st.write("--------")

st.subheader("Dataset Details")


st.write(f"""
         
        {
            '\n'.join(boston.DESCR.splitlines()[29:42])
        }
             
        {
            '\n'.join(boston.DESCR.splitlines()[11:21])[1:]
        }
            
            """)



X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target,columns = boston.target_names)
st.write("Features")
X
st.write("Target")
Y

st.sidebar.header("Specify Input Parameter")

def user_input_features():
    medInc = st.sidebar.slider('Median Income',X.MedInc.min(),X.MedInc.max(),X.MedInc.mean())
    
    houseAge = st.sidebar.slider('HouseAge',X.HouseAge.min(),X.HouseAge.max(),X.HouseAge.mean())
    
    aveRooms = st.sidebar.slider("Average Rooms",X.AveRooms.min(),X.AveRooms.max(),X.AveRooms.median())
    
    aveBedrms = st.sidebar.slider("Average Bedrooms",X.AveBedrms.min(),X.AveBedrms.max(),X.AveBedrms.median())
    
    Popualtion = st.sidebar.slider("Average Occupation",X.Population.min(),X.Population.max(),X.Population.median())
    AveOccup =st.sidebar.slider("Average ",X.AveOccup.min(),X.AveOccup.max(),X.AveOccup.median())
    Latitude = st.sidebar.slider("Average Rooms",X.Latitude.min(),X.Latitude.max(),X.Latitude.median())
    Longitude = st.sidebar.slider("Average Rooms",X.Longitude.min(),X.Longitude.max(),X.Longitude.median())
    
    data ={
        'MedInc':medInc,
        'HouseAge':houseAge,
        'AveRooms':aveRooms,
        'AveBedrms':aveBedrms,
        'Population':Popualtion,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude   
    }
    
    features = pd.DataFrame(data,index=[0])
    return features
    

st.write('---')

    
#Input Parameters
st.subheader("Specified Input Parameters")  
df = user_input_features()
st.write(df)
st.write('---')

#Build model
model = pickle.load(open('calif_house.pkl','rb'))

#apply model to make prediction
prediction = model.predict(df)

st.subheader('Prediction of MEDV')
st.write(prediction)
st.write('---')
st.balloons()


#Visualization

features = st.multiselect("Select features",X.columns,default=X.columns.tolist())
target = str(boston.target_names[0])
st.write(f"Target: {target}")

@st.cache_data
def plots():
    #Scatter Plot
    if features:
        
        st.write("Scatter Plot")
        fig,ax = plt.subplots()
        for feature in features:
            sns.scatterplot(x=X[feature],y=Y[target],ax=ax,label=feature)
        ax.set_xlabel("Features")
        ax.set_ylabel("Median House value")
        ax.legend()
        st.pyplot(fig)
        
        #Pair plot
        st.write("## Pair Plot")
        pair_plot = sns.pairplot(X+Y)
        data = pd.concat([X,Y],axis=1)
        pair_plot = sns.pairplot(data[features+[target]])
        st.pyplot(pair_plot)
        
        
        #Regression line plot
        st.write("### Regression Line Plot")
        for feature in features:
            fig, ax = plt.subplots()
            sns.regplot(x=data[feature], y=data[target], ax=ax, label=feature)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.legend()
            st.pyplot(fig)
            
plots()
        
    



    







    
    
