import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL.Image import Image
import keras.models
from keras.models import load_model

# Используем функции рекомендаций по типу погоды
from recommender import get_songs_by_weather, get_movies_by_weather

# Импортируем нейромодели для предсказания погоды
temp_model = keras.models.load_model("temp_full.keras")
weather_model = keras.models.load_model("weather_full.keras")
weather_encoding = {"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4}
weather_encoding_ru = {"морось": 0, "дождь": 1, "солнце": 2, "снег": 3, "туман": 4}
weather_encoding_ru_text = [':violet[пасмурную:cloud:]', ':blue[дождливую:rain_cloud:]', ':orange[солнечную:sunny:]', ':blue[снежную:snowflake:]', ':gray[туманную:fog:]']
HISTORY=4

def main():

    st.image("assets/mascot.png", width=180)
    st.title('Привет, я :violet[Погодный Советник]')
    st.header("Я предскажу погоду:sun_behind_cloud:, подскажу фильмы и песни:clapper::musical_note:")

    # Основная форма для получения вводных данных
    with st.form(key="main_form"):
        
        # Получение предыдущей температуры
        st.subheader('Какая погода была в предыдущие дни:calendar:?')
        st.caption('Напиши температуру предыдущих дней:thermometer:')
        temp_cols = st.columns(HISTORY)
        temp_inputs = []
        for i, col in enumerate(temp_cols):
            temp_inputs.append(col.text_input(f'День #{i+1}', value=f'{15+i*2}'))
    
        # Получение предыдущих типов погоды 
        st.caption('А какая была погода:thinking_face:?')
        weather_cols = st.columns(HISTORY)
        weather_inputs = []
        for i, col in enumerate(weather_cols):
            weather_inputs.append(col.selectbox(f'День #{i+1}',('солнце', 'морось', 'дождь','снег','туман')))
    
        # Кнопка отправки данных
        submit_button = st.form_submit_button(label="Предсказать:sparkles:")

    if submit_button:
        # Получение данных температуры
        temp_inputs = [float(x) for x in temp_inputs]
        temp_inputs = np.expand_dims(temp_inputs, axis=0)
        # Предсказание температуры с использованием нейромодели
        temp_result=temp_model.predict(temp_inputs)

        # Получение типов погоды
        # Перевод в кодовые значения
        coded_weather = [weather_encoding_ru[j] for j in weather_inputs]
        coded_weather = np.array(coded_weather)
        coded_weather = np.expand_dims(coded_weather, axis=0)
        # Предсказание типа погоды с использованием нейромодели
        predicted_weather_index = np.argmax(weather_model(coded_weather))
        predicted_weather = list(weather_encoding.keys())[predicted_weather_index]

        # Вывод полученных предсказаний
        st.subheader(f'Я думаю, что градусник покажет примерно {int(temp_result[0][0])}°C, а за окном будет {list(weather_encoding_ru.keys())[predicted_weather_index]}')
        st.subheader(f'А вот что можно посмотреть и послушать в {weather_encoding_ru_text[predicted_weather_index]} погоду')
        container = st.container(border=True)
        with container:
            st.subheader(f"Рекомендуемые фильмы")
            # Генерация рекомендованных фильмов
            recommended_movies = get_movies_by_weather("Weather", "Description", predicted_weather, 25)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f'**{recommended_movies.loc[0, "Title"]}**')
                st.write(recommended_movies.loc[0, "Genre"])
                st.write(recommended_movies.loc[0, "Rating"])
                st.write(recommended_movies.loc[0, "Description"])

            with col2:
                st.write(f'**{recommended_movies.loc[1, "Title"]}**')
                st.write(recommended_movies.loc[1, "Genre"])
                st.write(recommended_movies.loc[1, "Rating"])
                st.write(recommended_movies.loc[1, "Description"])

            with col3:
                st.write(f'**{recommended_movies.loc[2, "Title"]}**')
                st.write(recommended_movies.loc[2, "Genre"])
                st.write(recommended_movies.loc[2, "Rating"])
                st.write(recommended_movies.loc[2, "Description"])

        with container:
            st.subheader(f"Рекомендуемые песни")
            # Генерация рекомендованных песен
            recommended_songs = get_songs_by_weather("Weather", "Track Name", predicted_weather, 25)
            # st.dataframe(recommended_songs)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f'**{recommended_songs.loc[0, "Track Name"]}**')
                st.write(recommended_songs.loc[0, "Artist"])
                st.image(recommended_songs.loc[0, "Image"], use_column_width=True)

            with col2:
                st.write(f'**{recommended_songs.loc[1, "Track Name"]}**')
                st.write(recommended_songs.loc[1, "Artist"])
                st.image(recommended_songs.loc[1, "Image"], use_column_width=True)

            with col3:
                st.write(f'**{recommended_songs.loc[2, "Track Name"]}**')
                st.write(recommended_songs.loc[2, "Artist"])
                st.image(recommended_songs.loc[2, "Image"], use_column_width=True)
        

    


if __name__ == "__main__":
    main()
