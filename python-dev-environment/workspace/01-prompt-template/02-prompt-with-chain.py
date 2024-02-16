import os

from langchain_community.llms import ollama

from langchain.prompts import PromptTemplate
#from langchain.schema import StrOutputParser

from langchain_core.output_parsers import StrOutputParser

ollama_base_url = os.getenv("OLLAMA_BASE_URL")

model = ollama.Ollama(
    base_url=ollama_base_url, 
    model='tinydolphin',
)

something = "golang"

# Prompt template
prompt = PromptTemplate.from_template(
    "Explain me in one sentence what is {something}"
)

# Chain using model and formatting          
chain = prompt | model | StrOutputParser()    

response = chain.invoke({"something": something})  
# /api/generate

print(response)
