import streamlit as st
import pickle
reg_model = pickle.load(open("boston_prediction.pkl","rb"))
scaler = pickle.load(open("scaling.pkl","rb"))


# 
st.header("House prediction")
# CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	pric
col1,col2,col3,col4 = st.columns(4)
with col1:
    CRIM = st.slider("crim",min_value = 0.0,max_value = 1.0,step = 0.01)
with col2:
     ZN = st.number_input("zn",min_value = 0, max_value = 50,step = 5)
with col3:     
    INDUS = st.slider("indus",min_value = 0,max_value = 20,step = 1)
with col4:
    CHAS = st.number_input("chas",min_value = 0, max_value = 50,step = 5)
col5,col6,col7,col8 = st.columns(4)
with col5:
    NOX = st.slider("nox",min_value = 0.0,max_value = 1.0,step = 0.01)
with col6:
     RM = st.number_input("rm",min_value = 0, max_value = 50,step = 5)
with col7:
    AGE = st.slider("age",min_value = 1,max_value = 100,step = 1)
with col8:
    DIS = st.number_input("dis",min_value = 0.0, max_value = 20.0,step = 1.0)
col9,col10,col11,col12,col13 = st.columns(5)
with col9:
    RAD = st.number_input("rad", min_value=0, max_value=20, step=1)
with col10:
    TAX = st.number_input("tax", min_value=100, max_value=1000, step=50)
with col11:
    PTRATIO	= st.number_input("PTRATIO", min_value=0.0, max_value=20.0, step=1.0)
with col12:
    B = st.number_input("b", min_value=50.0, max_value=1000.0, step=1.0)
with col13:
    LSTAT = st.number_input("lsat", min_value=0.0, max_value=20.0, step=1.0)

lis = [CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]
final_input = scaler.transform([lis])
output = reg_model.predict(final_input)[0]
if st.button("predict price"):
    st.write(output)
# da = scaler.fit\

#     ([lis])




