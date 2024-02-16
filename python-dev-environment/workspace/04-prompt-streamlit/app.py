import os

from langchain_community.llms import ollama

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

import streamlit as st

ollama_base_url = os.getenv("OLLAMA_BASE_URL")

model = ollama.Ollama(
    base_url=ollama_base_url, 
    model='tinydolphin',
)

# StreamLit weapp title
st.title("Tutorial Generator")

#programming_language = "python"

# Text input field for the user
programming_language = st.text_input("Choose a language to generate a tutorial")

# Prompt template
language_prompt = PromptTemplate.from_template(
    "Explain me in one sentence what is {programming_language}"
)
tutorial_prompt = PromptTemplate.from_template(
    """
    You are a developer, given the programming language defintion, 
    you will write a tutorial for the noobs about this language,
    with some source code examples.
    Language: {language_defintion}
    """
)
language_chain = language_prompt | model | StrOutputParser()
tutorial_chain = tutorial_prompt | model | StrOutputParser()

chain =(
    {"language_defintion": language_chain}
    | RunnablePassthrough.assign(tutorial=tutorial_chain)
)

if programming_language:
    with st.spinner("Generating a tutorial..."):          
        response = chain.invoke({"programming_language": programming_language})  
        st.header(response["language_defintion"])
        st.write(response["tutorial"])
