import os

from langchain_community.llms import ollama

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import streamlit as st

ollama_base_url = os.getenv("OLLAMA_BASE_URL")

model = ollama.Ollama(
    base_url=ollama_base_url, 
    model='tinydolphin',
)

# StreamLit weapp title
st.title("What is it?")

# Text input field for the user
something = st.text_input("Type a word")
#something = "golang"

# Prompt template
prompt = PromptTemplate.from_template(
    "Explain me in one sentence what is {something}"
)

# Executing the chain when the user has entered a word  
if something:
    # Display a spinner    
    with st.spinner("Computing ..."):          
        # Chain using model and formatting          
        chain = prompt | model | StrOutputParser()          
        # Invoking the chain      
        response = chain.invoke({"something": something})          
        # Displaying the response          
        st.write(response)

