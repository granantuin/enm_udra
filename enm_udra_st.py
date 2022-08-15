import numpy as np
import pandas as pd
from meteogaliciamodel import get_meteogalicia_model_4Km,get_meteogalicia_model_1Km
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid


st.set_page_config(page_title="ENM Platforma tres",layout="wide")

st.write("#### **Mapa situación estación meteorológica cabo Udra y puntos modelo WRF (1 Km) Meteogalicia**") 

#load algorithm file gust
algo_g_d0 = pickle.load(open("algorithms/gust_UDR_d0.al","rb"))
algo_g_d1 = pickle.load(open("algorithms/gust_UDR_d1.al","rb"))

#load raw meteorological model and get model variables
meteo_model = get_meteogalicia_model_1Km(algo_g_d0["coor"])

#map
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(algo_g_d0["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)

#Select meteorological model wind features
w_g0=(meteo_model[0:48].wind_gust0*1.94384).round(0).to_numpy()
dir0=(meteo_model[0:48].dir0).round(0).to_numpy()

#select x _var
model_x_var_g_d0 = meteo_model[:24][algo_g_d0["x_var"]]
model_x_var_g_d1 = meteo_model[24:48][algo_g_d1["x_var"]]

#forecast machine learning  gust knots
gust_ml_d0 = (algo_g_d0["pipe"].predict(model_x_var_g_d0)*1.94384).round(0)
gust_ml_d1 = (algo_g_d1["pipe"].predict(model_x_var_g_d1)*1.94384).round(0)

#load algorithm file dir
algo_dir_d0 = pickle.load(open("algorithms/dir_UDR_d0.al","rb"))
algo_dir_d1 = pickle.load(open("algorithms/dir_UDR_d1.al","rb"))

#select x _var
model_x_var_dir_d0 = meteo_model[:24][algo_dir_d0["x_var"]]
model_x_var_dir_d1 = meteo_model[24:48][algo_dir_d1["x_var"]]

#forecast machine learning wind direction degrees
dir_ml_d0 = algo_dir_d0["pipe"].predict(model_x_var_dir_d0)
dir_ml_d1 = algo_dir_d1["pipe"].predict(model_x_var_dir_d1)

#compare results
df_show=pd.DataFrame({"Hora UTC":meteo_model[:48].index,
                      "Machine Learning dirección grados": np.concatenate((dir_ml_d0,dir_ml_d1),axis=0),
                      "Modelo WRF dirección grados": dir0,
                      "Machine Learning racha máxima nudos": np.concatenate((gust_ml_d0,gust_ml_d1),axis=0),
                      "Modelo WRF racha máxima nudos":w_g0,
                      })
                     
st.title(""" Pronóstico viento en estación cabo Udra Modelo WRF de Meteogalicia y Machine Learning""")
st.write("###### **Dirección viento medio hora anterior (T-1hora, T]**")
st.write("###### **Racha máxima hora anterior (T-1hora, T]**")
AgGrid(df_show)

# link to actual Udra station data
today_s=pd.to_datetime("today").strftime("%d/%m/%Y)")
st.write("Estación Udra [link](https://www.meteogalicia.gal/observacion/meteovisor/indexChartDezHoxe.action?idEstacion=10905&dataSeleccionada="+today_s)

#download quality report
with open("reports/Informe_wind.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Descargar informe de calidad viento",
                    data=PDFbyte,
                    file_name="informe_wind.pdf",
                    mime='application/octet-stream')

#Precipitation
#load algorithm file precipitation marin d0 d1
algo_prec_d0=pickle.load(open("algorithms/prec_ENM_d0.al","rb"))
algo_prec_d1=pickle.load(open("algorithms/prec_ENM_d1.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model_4Km(algo_prec_d1["coor"])

#map
st.write("#### **Mapa situación ENM y puntos modelo WRF (4 Km) Meteogalicia**")
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(algo_prec_d1["coor"], hover_data=['distance'],
                           lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)

#select x _var
model_x_var_p0=meteo_model[:24][algo_prec_d0["x_var"]]
model_x_var_p1=meteo_model[24:48][algo_prec_d1["x_var"]]

#forecast machine learning precipitation
prec_ml0=algo_prec_d0["pipe"].predict_proba(model_x_var_p0)
prec_ml1=algo_prec_d1["pipe"].predict_proba(model_x_var_p1)

#show results
df_show_pre=pd.DataFrame(np.concatenate((prec_ml0,prec_ml1),axis=0),
                         columns=["no p","probabilidad de precipitación machine learning"])
df_show_pre["Hora UTC"]=meteo_model.index[0:48]
df_show_pre["Modelo WRF en punto más cercano"]=np.around(meteo_model[:48].prec0.values,decimals=1)
df_show_pre=df_show_pre.drop(columns=["no p"])
df_show_pre['probabilidad de precipitación machine learning'] = df_show_pre['probabilidad de precipitación machine learning'].map("{:.0%}".format)
st.title(""" Probabilidad de precipitación ENM con Modelo WRF y Machine Learning""")
st.write("###### **Probabilidad de precipitación hora anterior**")
AgGrid(df_show_pre)

st.write("Estación Marin [link](https://www.meteogalicia.gal/observacion/meteovisor/indexChartDezHoxe.action?idEstacion=14005&dataSeleccionada="+today_s)

#download quality report
with open("reports/informe_prec.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Descargar informe de calidad precipitación",
                    data=PDFbyte,
                    file_name="informe_prec.pdf",
                    mime='application/octet-stream')
