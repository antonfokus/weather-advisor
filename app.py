import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL.Image import Image
import keras.models
from keras.models import load_model

from recommender import get_songs_by_weather, get_movies_by_weather

temp_model = keras.models.load_model("temp_full.keras")
weather_model = keras.models.load_model("weather_full.keras")
weather_encoding = {"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4}
weather_encoding_ru = {"морось": 0, "дождь": 1, "солнце": 2, "снег": 3, "туман": 4}


weather_images = {
    "sun": "assets/img_sun.jpeg",
    "rain": "assets/img_rain.jpeg",
    "drizzle": "assets/img_drizzle.jpeg",
    "fog": "assets/img_fog.jpeg",
    "snow": "assets/img_snow.jpeg",
}

weather_colors = {
    "sun": "#E6F4F1",
    "rain": "#D8E6ED",
    "drizzle": "#FAF8FF",
    "snow": "#F3FAFF", 
    "fog": "#F4F9FF", 
}



def main():
    

    # # body_style()
    # left_co, cent_co, last_co = st.columns(3)
    # with cent_co:
    #     st.image("assets/logo.png")
    st.title("Погодный Советник")
    st.header("Предскажет погоду, подскажет фильмы и песни!")

    
    with st.form(key="main_form"):
        
        # Create columns for temperature inputs
        temp_cols = st.columns(10)
        temp_inputs = []
        for i, col in enumerate(temp_cols):
            temp_inputs.append(col.text_input(f't° в день #{i+1}', value=f'{15+i*2}'))
    
        # Create columns for weather type inputs
        weather_cols = st.columns(10)
        weather_inputs = []
        for i, col in enumerate(weather_cols):
            weather_inputs.append(col.selectbox(f'Погода в день #{i+1}',('sun', 'drizzle', 'rain','snow','fog')))
    
        # Submit button
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        temp_inputs = [float(x) for x in temp_inputs]
        temp_inputs = np.expand_dims(temp_inputs, axis=0)
        temp_result=temp_model.predict(temp_inputs)

        coded_weather = [weather_encoding[j] for j in weather_inputs]
        coded_weather = np.array(coded_weather)
        coded_weather = np.expand_dims(coded_weather, axis=0)
        predicted_weather_index = np.argmax(weather_model.predict(coded_weather))
        predicted_weather = list(weather_encoding.keys())[predicted_weather_index]

        st.subheader(f'Я думаю, что градусник покажет примерно {temp_result}°C, а за окном будет {weather_encoding_ru[predicted_weather_index]}')
        st.subheader(f"Рекомендации для погоды типа {weather}")
        container = st.container()
        with container:
            st.subheader(f"Рекомендуемые фильмы")
            recommended_movies = get_movies_by_weather("Weather", "Description", predicted_weather, 20)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(recommended_movies.loc[0, "Title"])
                st.write(recommended_movies.loc[0, "Genre"])
                st.write(recommended_movies.loc[0, "Rating"])
                st.write(recommended_movies.loc[0, "Description"])

            with col2:
                st.write(recommended_movies.loc[1, "Title"])
                st.write(recommended_movies.loc[1, "Genre"])
                st.write(recommended_movies.loc[1, "Rating"])
                st.write(recommended_movies.loc[1, "Description"])

            with col3:
                st.write(recommended_movies.loc[2, "Title"])
                st.write(recommended_movies.loc[2, "Genre"])
                st.write(recommended_movies.loc[2, "Rating"])
                st.write(recommended_movies.loc[2, "Description"])

        with container:
            st.subheader(f"Рекомендуемые песни")
            recommended_songs = get_songs_by_weather("Weather", "Track Name", predicted_weather, 20)
            # st.dataframe(recommended_songs)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(recommended_songs.loc[0, "Track Name"])
                st.image(recommended_songs.loc[0, "Image"], use_column_width=True)
                st.write(recommended_songs.loc[0, "Album"])

            with col2:
                st.write(recommended_songs.loc[1, "Track Name"])
                st.image(recommended_songs.loc[1, "Image"], use_column_width=True)
                st.write(recommended_songs.loc[1, "Album"])

            with col3:
                st.write(recommended_songs.loc[2, "Track Name"])
                st.image(recommended_songs.loc[2, "Image"], use_column_width=True)
                st.write(recommended_songs.loc[2, "Album"])
        

    #№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    # st.subheader('Предсказание температуры')
    # temps = [st.text_input(f'Температура в день #{i+1}', value=f'{np.random.randint(10, 20)}') for i in range(10)]
    # temps = [float(x) for x in temps]
    # if st.button("Предсказать температуру"):
    #     temps = np.expand_dims(temps, axis=0)
    #     st.write(temp_model.predict(temps))

    # st.subheader('Предсказание типа погоды')
    # weathers = [st.selectbox(f'Погода в день #{i+1}',('sun', 'drizzle', 'rain','snow','fog')) for i in range(10)]
    # if st.button("Предсказать погоду"):
    #     coded_weather = [weather_encoding[j] for j in weathers]
    #     coded_weather = np.array(coded_weather)
    #     coded_weather = np.expand_dims(coded_weather, axis=0)
    #     predicted_weather_index = np.argmax(weather_model.predict(coded_weather))
    #     predicted_weather = list(weather_encoding.keys())[predicted_weather_index]
    #     st.write(predicted_weather)

    #  st.subheader('Рекомендации фильмов и песен')
    # weather = st.selectbox(
    # 'Выберите вариант',
    # ('sun', 'drizzle', 'rain', 'snow','fog'))
    # if st.button("Получить рекомендации"):
    #     if weather:
    #         background_image = weather_images.get(weather)
    #         weather_color = weather_colors.get(weather)
    #         if background_image:
    #             st.markdown(
    #                 f"""
    #                 <style>
    #                 h3 {{
    #                     color: black;
                    
    #                 }}
                    
    #                 [data-testid="stAppViewContainer"]{{
    #                     background: '{background_image}';
    #                     background-size: cover;
    #                 }}
                    
    #                 div.css-ocqkz7.e1tzin5v3 {{
    #                     background-color: {weather_color};
    #                     border: 2px solid #CCCCCC;
    #                     padding: 5% 5% 5% 10%;
    #                     border-radius: 5px;
    #                     color: black;
    #                     text-align: center;
    #                     display: flex;
    #                     box-shadow: 0 3px 10px rgb(0 0 0 / 0.2);
    #                 }}
    #                 </style>
    #                 """,
    #                 unsafe_allow_html=True
    #             )
    #         st.subheader(f"Рекомендации для погоды типа {weather}")
    #         container = st.container()
    #         with container:
    #             st.subheader(f"Рекомендуемые фильмы")
    #             recommended_movies = get_movies_by_weather("Weather", "Description", weather, 20)
    #             col1, col2, col3 = st.columns(3)
    #             with col1:
    #                 st.write(recommended_movies.loc[0, "Title"])
    #                 st.write(recommended_movies.loc[0, "Genre"])
    #                 st.write(recommended_movies.loc[0, "Rating"])
    #                 st.write(recommended_movies.loc[0, "Description"])

    #             with col2:
    #                 st.write(recommended_movies.loc[1, "Title"])
    #                 st.write(recommended_movies.loc[1, "Genre"])
    #                 st.write(recommended_movies.loc[1, "Rating"])
    #                 st.write(recommended_movies.loc[1, "Description"])

    #             with col3:
    #                 st.write(recommended_movies.loc[2, "Title"])
    #                 st.write(recommended_movies.loc[2, "Genre"])
    #                 st.write(recommended_movies.loc[2, "Rating"])
    #                 st.write(recommended_movies.loc[2, "Description"])

    #         with container:
    #             st.subheader(f"Рекомендуемые песни")
    #             recommended_songs = get_songs_by_weather("Weather", "Track Name", weather, 20)
    #             # st.dataframe(recommended_songs)
    #             col1, col2, col3 = st.columns(3)
    #             with col1:
    #                 st.write(recommended_songs.loc[0, "Track Name"])
    #                 st.image(recommended_songs.loc[0, "Image"], use_column_width=True)
    #                 st.write(recommended_songs.loc[0, "Album"])

    #             with col2:
    #                 st.write(recommended_songs.loc[1, "Track Name"])
    #                 st.image(recommended_songs.loc[1, "Image"], use_column_width=True)
    #                 st.write(recommended_songs.loc[1, "Album"])

    #             with col3:
    #                 st.write(recommended_songs.loc[2, "Track Name"])
    #                 st.image(recommended_songs.loc[2, "Image"], use_column_width=True)
    #                 st.write(recommended_songs.loc[2, "Album"])
    #     else:
    #         st.warning("Пожалуйста, выберите тип погоды.")



if __name__ == "__main__":
    main()
