import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model_4Km, get_meteogalicia_model_1Km, get_table_download_link
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid
import matplotlib.pyplot as plt

st.set_page_config(page_title="ENM_UDRA",layout="wide")

#load algorithm file gust
algo_g_d0 = pickle.load(open("algorithms/gust_UDR_d0.al","rb"))
algo_g_d1 = pickle.load(open("algorithms/gust_UDR_d1.al","rb"))
algo_g_d2 = pickle.load(open("algorithms/gust_UDR_d2.al","rb"))

#load raw meteorological model and get model variables
meteo_model = get_meteogalicia_model_1Km(algo_g_d0["coor"])

#Select meteorological model wind features
w_g0 = (meteo_model[0:72].wind_gust0*1.94384).round(0).to_numpy()
dir0 = (meteo_model[0:72].dir0).round(0).to_numpy()
mod0 = (meteo_model[0:72].dir0).round(0).to_numpy()

#select x _var
model_x_var_g_d0 = meteo_model[:24][algo_g_d0["x_var"]]
model_x_var_g_d1 = meteo_model[24:48][algo_g_d1["x_var"]]
model_x_var_g_d2 = meteo_model[48:72][algo_g_d2["x_var"]]

#forecast machine learning  gust knots
gust_ml_d0 = (algo_g_d0["pipe"].predict(model_x_var_g_d0)*1.94384).round(0)
gust_ml_d1 = (algo_g_d1["pipe"].predict(model_x_var_g_d1)*1.94384).round(0)
gust_ml_d2 = (algo_g_d2["pipe"].predict(model_x_var_g_d2)*1.94384).round(0)

#load algorithm file dir
algo_dir_d0 = pickle.load(open("algorithms/dir_UDR_d0.al","rb"))
algo_dir_d1 = pickle.load(open("algorithms/dir_UDR_d1.al","rb"))
algo_dir_d2 = pickle.load(open("algorithms/dir_UDR_d2.al","rb"))

#select x _var
model_x_var_dir_d0 = meteo_model[:24][algo_dir_d0["x_var"]]
model_x_var_dir_d1 = meteo_model[24:48][algo_dir_d1["x_var"]]
model_x_var_dir_d2 = meteo_model[48:72][algo_dir_d2["x_var"]]

#forecast machine learning wind direction degrees
dir_ml_d0 = algo_dir_d0["pipe"].predict(model_x_var_dir_d0)
dir_ml_d1 = algo_dir_d1["pipe"].predict(model_x_var_dir_d1)
dir_ml_d2 = algo_dir_d2["pipe"].predict(model_x_var_dir_d2)

#load algorithm file spd
algo_spdb_d0 = pickle.load(open("algorithms/spd_udr_1km_d0.al","rb"))
algo_spdb_d1 = pickle.load(open("algorithms/spd_udr_1km_d1.al","rb"))
algo_spdb_d2 = pickle.load(open("algorithms/spd_udr_1km_d2.al","rb"))

#select x _var
model_x_var_spdb_d0 = meteo_model[:24][algo_spdb_d0["x_var"]]
model_x_var_spdb_d1 = meteo_model[24:48][algo_spdb_d1["x_var"]]
model_x_var_spdb_d2 = meteo_model[48:72][algo_spdb_d2["x_var"]]

#forecast machine learning wind intensity Beaufort
spdb_ml_d0 = algo_spdb_d0["pipe"].predict(model_x_var_spdb_d0)
spdb_ml_d1 = algo_spdb_d1["pipe"].predict(model_x_var_spdb_d1)
spdb_ml_d2 = algo_spdb_d2["pipe"].predict(model_x_var_spdb_d2)


#compare results
df_show=pd.DataFrame({"ML dir": np.concatenate((dir_ml_d0,dir_ml_d1,dir_ml_d2),axis=0),
                      "ML spdb" : np.concatenate((dir_ml_d0,dir_ml_d1,dir_ml_d2),axis=0),
                      "WRF dir": dir0,
                      "WRF spd": mod0,
                      "Hora UTC":meteo_model[:72].index,
                      "ML racha": np.concatenate((gust_ml_d0,gust_ml_d1,gust_ml_d2),axis=0),
                      "WRF racha":w_g0,
                      })
              

st.write("#### **Pronóstico viento en estación cabo Udra Modelo WRF de Meteogalicia y Machine Learning**")
st.write("###### **Dirección viento medio hora anterior (grados)**")

#AgGrid(df_show)

#label wrf direction
interval = pd.IntervalIndex.from_tuples([(-0.5,20), (20, 40), (40, 60),
                                           (60,80),(80,100),(100,120),(120,140),(140,160),
                                           (160,180),(180,200),(200,220),(220,240),
                                           (240,260),(260,280),(280,300),(300,320),
                                           (320,340),(340,360)])
labels = ['[0, 20]', '(20, 40]', '(40, 60]','(60, 80]', '(80, 100]',
          '(100, 120]', '(120, 140]','(140, 160]', '(160, 180]', '(180, 200]',
          '(200, 220]','(220, 240]', '(240, 260]', '(260, 280]', '(280, 300]',
          '(300, 320]', '(320, 340]', '(340, 360]']
df_show["dir_WRF_l"] = pd.cut(df_show["WRF dir"], bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)

#label wrf spd to Beaufort scale
labels = ["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]
interval = pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),
                                     (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),
                                     (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])
df_show["spd_WRF_l"] = pd.cut(df_show["WRF spd"], bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)

#show results wind direction
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_show["Hora UTC"], df_show['ML dir'], marker="^", color="b",markersize=8, 
         markerfacecolor='w', linestyle='')
plt.plot(df_show["Hora UTC"], df_show['dir_WRF_l'], color="r",marker="v", markersize=8,
         markerfacecolor='w', linestyle='');
plt.legend(('dirección ml', 'dirección WRF'),)
plt.grid(True)
plt.title("Modelo meteorológico WRF (precisión 26%) versus machine learning (precisión 46%)")
st.pyplot(fig)

#probabilistic results
prob = (np.concatenate((algo_dir_d0["pipe"].predict_proba(model_x_var_dir_d0),
                        algo_dir_d1["pipe"].predict_proba(model_x_var_dir_d1),
                        algo_dir_d2["pipe"].predict_proba(model_x_var_dir_d2)),
                       axis =0)).transpose()
df_prob = pd.DataFrame(prob,index = (algo_dir_d0["pipe"].classes_ )).T

# Find the columns where all values are less than or equal to 5%
cols_to_drop = df_prob.columns[df_prob.apply(lambda x: x <= 0.05).all()]
df_prob.drop(cols_to_drop, axis=1, inplace=True)
df_prob["time"] = meteo_model[:72].index

st.write("""Probabilidades dirección del viento columnas con más del 5%""")
AgGrid(round(df_prob,2))


#show results wind gust
st.write("###### **Racha máxima hora anterior (nudos)**")
fig, ax = plt.subplots(figsize=(10,6))
df_show.set_index("Hora UTC")[["ML racha","WRF racha"]].plot(grid=True, ax=ax, color=["b","r"], linestyle='--');
ax.set_title("Modelo meteorologico WRF (MAE=2.1) versus machine learning (MAE =1.4)")
st.pyplot(fig)


#Download excel file
st.markdown(get_table_download_link(df_show),unsafe_allow_html=True)

st.write("###### **Información complementaria:**")
#map
if st.checkbox("Mapa situación estación meteorológica cabo Udra y puntos modelo WRF (1 Km) Meteogalicia"):
  px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
  dist_map=px.scatter_mapbox(algo_g_d0["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                             color_continuous_scale=px.colors.cyclical.IceFire,)
  st.plotly_chart(dist_map)

#link to actual Udra station data
today_s=pd.to_datetime("today").strftime("%d/%m/%Y)")
st.write("Estación Udra en tiempo real [enlace](https://www.meteogalicia.gal/observacion/meteovisor/indexChartDezHoxe.action?idEstacion=10905&dataSeleccionada="+today_s)

#download quality report
with open("reports/informe_wind.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Descargar informe de calidad viento",
                    data=PDFbyte,
                    file_name="informe_wind.pdf",
                    mime='application/octet-stream')



#Precipitation
#load algorithm file precipitation marin d0 d1
algo_prec_d0 = pickle.load(open("algorithms/prec_ENM_d0.al","rb"))
algo_prec_d1 = pickle.load(open("algorithms/prec_ENM_d1.al","rb"))
algo_prec_d2 = pickle.load(open("algorithms/prec_ENM_d2.al","rb"))
algo_prec_d3 = pickle.load(open("algorithms/prec_ENM_d3.al","rb"))

#load raw meteorological model and get model variables
meteo_model = get_meteogalicia_model_4Km(algo_prec_d1["coor"])

#select x _var
model_x_var_p0 = meteo_model[:24][algo_prec_d0["x_var"]]
model_x_var_p1 = meteo_model[24:48][algo_prec_d1["x_var"]]
model_x_var_p2 = meteo_model[48:72][algo_prec_d2["x_var"]]
model_x_var_p3 = meteo_model[72:96][algo_prec_d2["x_var"]]

#forecast machine learning precipitation
prec_ml0 = algo_prec_d0["pipe"].predict_proba(model_x_var_p0)
prec_ml1 = algo_prec_d1["pipe"].predict_proba(model_x_var_p1)
prec_ml2 = algo_prec_d2["pipe"].predict_proba(model_x_var_p2)
prec_ml3 = algo_prec_d3["pipe"].predict_proba(model_x_var_p3)

#show results
df_show_pre = pd.DataFrame(np.concatenate((prec_ml0,prec_ml1,prec_ml2,prec_ml3),axis=0),
                         columns=["no p","ML"])
df_show_pre["Hora UTC"] = meteo_model.index[0:96]
df_show_pre["WRF"] = np.around(meteo_model[:96].prec0.values,decimals=1)
df_show_pre = df_show_pre.drop(columns=["no p"])
#df_show_pre['ML'] = df_show_pre['ML'].map("{:.0%}".format)
df_show_pre['Hora UTC'] = pd.to_datetime(df_show_pre['Hora UTC'])

st.write("#### **Probabilidad de precipitación hora anterior con Machine Learning (Heidke Skill Score: 0.53) en ENM**")         
fig, ax = plt.subplots(figsize=(16,8))
df_show_pre.set_index('Hora UTC')["ML"].plot(ax=ax, grid=True, kind='bar')
st.pyplot(fig)

st.write("#### **Precipitación prevista en mm hora anterior con modelo meteorológico WRF(Heidke Skill Score: 0.41) en ENM**")         
fig, ax = plt.subplots(figsize=(16,8))
df_show_pre.set_index('Hora UTC')["WRF"].plot(ax=ax, grid=True, kind='bar')
st.pyplot(fig)

#download  excel file  
st.markdown(get_table_download_link(df_show_pre),unsafe_allow_html=True)

st.write("###### **Información complementaria:**")
#map
if st.checkbox("Mapa situación ENM y puntos modelo WRF (4 Km) Meteogalicia"):
  px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
  dist_map=px.scatter_mapbox(algo_prec_d1["coor"], hover_data=['distance'],
                             lat='lat', lon='lon',color='distance',
                             color_continuous_scale=px.colors.cyclical.IceFire,)
  st.plotly_chart(dist_map)
  
     
#link to actual  Marin station data
st.write("Estación Marín en tiempo real [enlace](https://www.meteogalicia.gal/observacion/meteovisor/indexChartDezHoxe.action?idEstacion=14005&dataSeleccionada="+today_s)

#download quality report
with open("reports/informe_prec.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Descargar informe de calidad precipitación",
                    data=PDFbyte,
                    file_name="informe_prec.pdf",
                    mime='application/octet-stream')
