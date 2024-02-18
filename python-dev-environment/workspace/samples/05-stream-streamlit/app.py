import os
# https://github.com/langchain-ai/langchain/issues/11398
from langchain_community.llms import ollama
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser

#from langchain_core.runnables import RunnablePassthrough

import streamlit as st


ollama_base_url = os.getenv("OLLAMA_BASE_URL")

model = ollama.Ollama(
    base_url=ollama_base_url, 
    model='llama2',
)

# StreamLit weapp title
st.title("Tell me more about this language?")

# Text input field for the user
language = st.text_input("Type the name of a programming language")
#something = "golang"

# Prompt template
prompt = PromptTemplate.from_template(
    "Make a short presentation of this programming language {language}"
)
#output = st.empty() 
# Executing the chain when the user has entered a word  
if language:
    # Display a spinner    
    #with st.spinner("Computing ..."):   
        st_callback = StreamlitCallbackHandler(st.container())
    
        # Chain using model and formatting          
        chain = prompt | model          
        # Invoking the chain      
        response = chain.invoke({"language": language}, {"callbacks": [st_callback]})     
        #output.markdown(response)
        # Displaying the response          
        #st.write(response)
    #st.write("done")


