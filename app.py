import requests
import streamlit as st
import pandas as pd
from PIL.Image import Image

from recommender import get_songs_by_weather, get_movies_by_weather

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



def body_style():
    st.markdown(
        f"""
        <style>
        #MainMenu, header, footer {{
            visibility: hidden;
        }}
        body {{
            background-color: #d3f2ef;
            font-family: 'Comfortaa', sans-serif !important;
        }}
        h1 {{
            color: #e6e6fa;
            
        }}
        label {{
            color: #e6e6fa;
            
        }}

        img {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }}

        .css-ffhzg2 {{
            background: rgb(37,29,57);
            background: -moz-radial-gradient(circle, rgba(37,29,57,1) 0%, rgba(44,45,69,1) 29%, rgba(57,74,90,1) 65%, rgba(76,118,123,1) 95%, rgba(76,118,123,1) 100%, rgba(96,165,158,1) 100%);
            background: -webkit-radial-gradient(circle, rgba(37,29,57,1) 0%, rgba(44,45,69,1) 29%, rgba(57,74,90,1) 65%, rgba(76,118,123,1) 95%, rgba(76,118,123,1) 100%, rgba(96,165,158,1) 100%);
            background: radial-gradient(circle, rgba(37,29,57,1) 0%, rgba(44,45,69,1) 29%, rgba(57,74,90,1) 65%, rgba(76,118,123,1) 95%, rgba(76,118,123,1) 100%, rgba(96,165,158,1) 100%);
            filter: progid:DXImageTransform.Microsoft.gradient(startColorstr="#251d39",endColorstr="#60a59e",GradientType=1);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():

    body_style()
    st.title("Погодный Советник")
    st.subheader("Предскажет погоду, подскажет фильмы и песни!")
    weather = st.selectbox(
    'Выберите вариант',
    ('drizzle', 'rain', 'sun', 'snow','fog'))
    if st.button("Получить рекомендации"):
        if city:
            weather = get_weather(city)
            if weather:
                background_image = weather_images.get(weather)
                weather_color = weather_colors.get(weather)
                if background_image:
                    st.markdown(
                        f"""
                        <style>
                        h3 {{
                            color: white;
                            text-shadow: black 0.2em 0.2em 0.2em;
                        }}
                        
                        [data-testid="stAppViewContainer"]{{
                            background: url('{background_image}');
                            background-size: cover;
                        }}
                        
                        div.css-ocqkz7.e1tzin5v3 {{
                            background-color: {weather_color};
                            border: 2px solid #CCCCCC;
                            padding: 5% 5% 5% 10%;
                            border-radius: 5px;
                            color: black;
                            text-align: center;
                            display: flex;
                            box-shadow: 0 3px 10px rgb(0 0 0 / 0.2);
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                st.subheader(f"Recommendations for {city} city in {weather}")
                container = st.container()
                with container:
                    st.subheader(f"Recommended Movies")
                    recommended_movies = get_movies_by_weather("Weather", "Description", weather, 3)
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
                    st.subheader(f"Recommended Songs")
                    recommended_songs = get_songs_by_weather("Weather", "Track Name", weather, 3)
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

            else:
                st.error("Failed to fetch weather information. Please check the city name.")
        else:
            st.warning("Please enter a city name.")
    else:
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image("assets/drWeather.png")


if __name__ == "__main__":
    main()
