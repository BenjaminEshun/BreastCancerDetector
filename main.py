from classifier import classify, set_bg_hack
import tensorflow as tf
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer App",
    page_icon="data/icon.ico",
    layout="centered", )


st.markdown("""
<style>
.css-zq5wmm.ezrtsby0
{
  visibility : hidden;
}
.css-h5rgaw.ea3mdgi1
{
  visibility : hidden;
}
<\style>
""", unsafe_allow_html=True)

# Setting background image
set_bg_hack("image.jpg")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title(":rainbow[BREAST CANCER DETECTOR]")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bc_model.hdf5")
    return model


with st.spinner('Loading Model Into Memory.....'):
    model = load_model()

file = st.file_uploader("", type=["jpg", "png", "jpeg"])

classify(file, model)
