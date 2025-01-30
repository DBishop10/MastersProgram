import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display the main title
st.markdown("# Self Check 4")

# Display the subtitle
st.markdown("## Edit the Camera View")

# Check if the DataFrame exists in session state, if not, initialize it so it shows up on load
if 'df' not in st.session_state:
    rows, cols = 4, 4  # Default values for rows and columns
    data = [[np.random.randint(0, 101)/100 for _ in range(cols)] for _ in range(rows)]
    st.session_state.df = pd.DataFrame(data)

# Side panel for selecting DataFrame dimensions with a submit button
with st.sidebar.form(key='grid_form'):
    st.markdown("## Grid Size")
    cols = st.number_input("Select a width", 2, 10, 4)
    rows = st.number_input("Select a height", 2, 10, 4)
    submit_button = st.form_submit_button(label='Submit')

# Generate the DataFrame when the form is submitted
if submit_button:
    data = [[np.random.randint(0, 101)/100 for _ in range(cols)] for _ in range(rows)]
    st.session_state.df = pd.DataFrame(data)

# Display the DataFrame
if not st.session_state.df.empty:
    edited_df = st.data_editor(st.session_state.df)
    st.session_state.df = edited_df

# Button to display the camera view
if st.button('Display') and not st.session_state.df.empty:
    df = st.session_state.df
    figure = plt.figure(figsize=(2, 2))
    axes = figure.add_subplot(1, 1, 1)
    axes.set_title("Camera View")

    # Convert DataFrame to pixel values
    pixels = np.array(255 - df.values * 255, dtype='uint8')
    # pixels = pixels.reshape((rows, cols))
    axes.imshow(pixels, cmap='gray')
    
    # Display the image
    st.pyplot(plt.gcf(), use_container_width=False)
