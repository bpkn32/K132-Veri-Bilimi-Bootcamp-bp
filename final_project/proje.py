import streamlit as st
import pandas as pd
import pickle


st.header("Meme Tümörünün İkili Sınıflandırılması")
st.subheader("Tahlil sonuçlarını giriniz:")
main_data = pd.read_csv("D:\Downloads\data.csv")
main_data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
main_data.columns = main_data.columns.map(lambda x : x.replace(" ","_"))


radius_mean = st.number_input("radius_mean",0.0,100.0,17.99,1e-6)
texture_mean = st.number_input("texture_mean",0.0,100.0,10.38,1e-6)
perimeter_mean = st.number_input("perimeter_mean",0.0,300.0,122.80,1e-10)
area_mean = st.number_input("area_mean",0.0,3000.0,1001.0,1e-10)
smoothness_mean = st.number_input("smoothness_mean",0.0,5.0,0.11840,1e-10)
compactness_mean = st.number_input("compactness_mean",0.0,5.0,0.27760,1e-10)
concavity_mean = st.number_input("concavity_mean",0.0,5.0,0.3001,1e-10)
concave_points_mean = st.number_input("concave_points_mean",0.0,5.0,0.14710,1e-10)
symmetry_mean = st.number_input("symmetry_mean",0.0,5.0,0.2419,1e-10)
fractal_dimension_mean = st.number_input("fractal_dimension_mean",0.0,5.0,0.07871,1e-10)
radius_se = st.number_input("radius_se",0.0,10.0,0.9053,1e-10)
texture_se = st.number_input("texture_se",0.0,100.0,0.9053,1e-10)
perimeter_se = st.number_input("perimeter_se",0.0,50.0,8.589,1e-10)
area_se = st.number_input("area_se",0.0,1000.0,153.4,1e-10)
smoothness_se = st.number_input("smoothness_se",0.0,5.0,0.006399,1e-10)
compactness_se = st.number_input("compactness_se",0.0,5.0,0.04904,1e-10)
concavity_se = st.number_input("concavity_se",0.0,5.0,0.05373,1e-10)
concave_points_se = st.number_input("concave_points_se",0.0,5.0,0.01587,1e-10)
symmetry_se = st.number_input("symmetry_se",0.0,1.0,0.03003,1e-10)
fractal_dimension_se = st.number_input("fractal_dimension_se",0.0,5.0,0.006193,1e-10)
radius_worst =st.number_input("radius_worst",0.0,100.0,25.38,1e-10)
texture_worst = st.number_input("texture_worst",0.0,100.0,17.33,1e-10)
perimeter_worst = st.number_input("perimeter_worst",0.0,500.0,184.6,1e-10)
area_worst = st.number_input("area_worst",0.0,10000.0,2019.0,1e-10)
smoothness_worst = st.number_input("smoothness_worst",0.0,5.0,0.1622,1e-10)
compactness_worst = st.number_input("compactness_worst",0.0,10.0,0.6656,1e-10)
concavity_worst =st.number_input("concavity_worst",0.0,10.0,0.7119,1e-10)
concave_points_worst = st.number_input("concave_points_worst",0.0,5.0,0.2654,1e-10)
symmetry_worst = st.number_input("symmetry_worst",0.0,5.0,0.4601,1e-10)
fractal_dimension_worst = st.number_input("fractal_dimension_worst",0.0,5.0,0.1189,1e-10)
data = {"radius_mean" :radius_mean,
        "texture_mean" :texture_mean,
        "perimeter_mean" :perimeter_mean,
        "area_mean" :area_mean,
        "smoothness_mean" :smoothness_mean,
        "compactness_mean" :compactness_mean,
        "concavity_mean" :concavity_mean,
        "concave_points_mean" :concave_points_mean,
        "symmetry_mean" :symmetry_mean,
        "fractal_dimension_mean" :fractal_dimension_mean,
        "radius_se" :radius_se,
        "texture_se" :texture_se,
        "perimeter_se" :perimeter_se,
        "area_se" :area_se,
        "smoothness_se" :smoothness_se,
        "compactness_se" :compactness_se,
        "concavity_se" :concavity_se,
        "concave_points_se" :concave_points_se,
        "symmetry_se" :symmetry_se,
        "fractal_dimension_se" :fractal_dimension_se,
        "radius_worst" :radius_worst,
        "texture_worst" :texture_worst,
        "perimeter_worst" :perimeter_worst,
        "area_worst" :area_worst,
        "smoothness_worst" :smoothness_worst,
        "compactness_worst" :compactness_worst,
        "concavity_worst" :concavity_worst,
        "concave_points_worst" :concave_points_worst,
        "symmetry_worst" :symmetry_worst,
        "fractal_dimension_worst" :fractal_dimension_worst}
features = pd.DataFrame(data=data,index = [0])

df = features
loaded_model = pickle.load(open("model.pkl","rb"))
prediction = loaded_model.predict(features)
prediction_proba = loaded_model.predict_proba(features)
st.subheader("Tahlil Sonuçları :")
st.write(df)

st.subheader("Tahmin :")
st.write("0 : 'M' (Kötü huylu)")
st.write("1 : 'B' (İyi huylu)")
st.write(prediction)

st.subheader("Tahminin Olasılığı :")
st.write(prediction_proba)
  