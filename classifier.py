import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64


def classify(file, model):
    if file is None:
        st.write(":red[Please upload a microscopic biopsy image to proceed]")

    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        img = tf.keras.utils.load_img(file, target_size=(400, 400))
        x = tf.keras.utils.img_to_array(img)
        x /= 255.0
        x = np.expand_dims(x, axis=0)

        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)

        # if st.button('**Run Model**'):
        if classes[0] > 0.5:
            st.markdown(
                f'<h1 style="color:#FF0000;font-size:30px;">{"This image has cancerous cells (Malignant)!"}</h1>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<h1 style="color:#33ff33;font-size:30px;">{"This image has no cancerous cells (Benign)"}</h1>',
                unsafe_allow_html=True)


def set_bg_hack(main_bg):
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
