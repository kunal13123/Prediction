import streamlit as st
import  pickle
import numpy as np
import pandas as pd

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

#car1 = pickle.load(open("Cleaned_Car_Data.pkl"))
car = pd.read_csv("Cleaned_Car_data.csv")
pipe1 = pickle.load(open("LinearRegressionModel.pkl",'rb'))

st.title("what do you want to predict")
scsc=st.selectbox('Choose any one',['Laptop','Car'])

if(scsc=='Laptop'):

            st.title("Laptop Predictor")

            company = st.selectbox('Brand',df['Company'].unique())

            type = st.selectbox('Type',df['TypeName'].unique())

            ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,640])

            weight = st.number_input('Weight of the laptop')

            touchscreen = st.selectbox('Touchscreen',['No','Yes'])

            ips = st.selectbox('IPS',['No','Yes'])

            screen_size = st.number_input('Screen_Size')

            resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])\

            cpu = st.selectbox('CPU',df['Cpu brand'].unique())

            hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

            ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

            gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

            os = st.selectbox('OS',df['os'].unique())

            if st.button('Predict Laptop Price'):
                ppi = None
                if touchscreen == 'Yes':
                    touchscreen = 1
                else:
                    touchscreen = 0

                if ips == 'Yes':
                    ips = 1
                else:
                    ips = 0

                x_res = int(resolution.split('x')[0])
                y_res = int(resolution.split('x')[1])
                ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
                query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

                query = query.reshape(1,12)
                st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))


if(scsc=='Car'):
                st.title("Car Predictor")

                companies = st.selectbox('Companies',sorted(car['company'].unique()))
                car_models = st.selectbox('Car Models', sorted(car['name'].unique()))
                year = st.selectbox('Year' , sorted(car['year'].unique(),reverse=True))
                fuel_type = st.selectbox('Fuel Type',car['fuel_type'].unique())
                kms_driven = st.number_input("Kms_Driven")

                if st.button('Predict Car Price'):
                    prediction=pipe1.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                         data=np.array([car_models,companies,year,kms_driven,fuel_type]).reshape(1, 5)))

                    st.title("The predicted price of this configuration is :" + str(int(prediction)))


