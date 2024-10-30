import torch
import gc
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import re

# Clear memory and CUDA cache
gc.collect()
torch.cuda.empty_cache()

# Title and headings
st.title("Personalized Diet and Workout Recommender:coffee:")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Your Best Food and Exercise Advisor:spoon:')
st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

# API key input
st.sidebar.header("API Key Setup")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if user_api_key:
    os.environ['OPENAI_API_KEY'] = user_api_key

# Validate if the API key is provided
if not os.getenv('OPENAI_API_KEY'):
    st.error("Please enter your OpenAI API key in the sidebar to proceed.")
else:
    # Define the LLM with the input key
    llm = OpenAI(temperature=0.9)

    # Prompt template
    prompt_template = PromptTemplate(
        input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'address', 'allergies'],
        template="Diet Recommendation System:\n"
                 "Please recommend 5 restaurants names, 5 breakfast names, 5 dinner names, and 5 workout names, "
                 "based on the following criteria given below:\n"
                 "Age: {age}\n"
                 "Gender: {gender}\n"
                 "Weight: {weight}\n"
                 "Height: {height}\n"
                 "Veg_or_Nonveg: {veg_or_nonveg}\n"
                 "Address: {address}\n"
                 "Food allergies: {allergies}."
    )

    # User input fields
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    weight = st.number_input("Weight (pounds)", min_value=0)
    height = st.number_input("Height (cm)", min_value=0)
    veg_or_nonveg = st.selectbox("Veg or Non-Veg", ["Veg", "Non-Veg"])
    address = st.text_input("Address")
    allergies = st.text_input("Food allergies")

    # Get Recommendations button
    if st.button("Get Recommendations"):
        chain = LLMChain(llm=llm, prompt=prompt_template)
        input_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'veg_or_nonveg': veg_or_nonveg,
            'address': address,
            'allergies': allergies
        }

        try:
            results = chain.run(input_data)

            # Extract recommendations
            restaurant_names = re.findall(r'Restaurants:(.*?)Breakfast:', results, re.DOTALL)
            breakfast_names = re.findall(r'Breakfast:(.*?)Dinner:', results, re.DOTALL)
            dinner_names = re.findall(r'Dinner:(.*?)Workouts:', results, re.DOTALL)
            workout_names = re.findall(r'Workouts:(.*?)$', results, re.DOTALL)

            # Display recommendations
            st.subheader("Recommendations")
            
            st.markdown("#### Restaurants")
            if restaurant_names:
                for restaurant in restaurant_names[0].strip().split('\n'):
                    st.write(restaurant.strip())
            else:
                st.write("No restaurant recommendations available.")

            st.markdown("#### Breakfast")
            if breakfast_names:
                for breakfast in breakfast_names[0].strip().split('\n'):
                    st.write(breakfast.strip())
            else:
                st.write("No breakfast recommendations available.")

            st.markdown("#### Dinner")
            if dinner_names:
                for dinner in dinner_names[0].strip().split('\n'):
                    st.write(dinner.strip())
            else:
                st.write("No dinner recommendations available.")

            st.markdown("#### Workouts")
            if workout_names:
                for workout in workout_names[0].strip().split('\n'):
                    st.write(workout.strip())
            else:
                st.write("No workout recommendations available.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
