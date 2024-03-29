import os

from langchain_community.llms import ollama
from langchain.prompts import PromptTemplate

ollama_base_url = os.getenv("OLLAMA_BASE_URL")

model = ollama.Ollama(
    base_url=ollama_base_url, 
    model='llama2',
)

prompt_template = PromptTemplate.from_template(
    "Explain {what} in {language}."
)
prompt = prompt_template.format(what="loop", language="python")

print(prompt + "\n")

completion = model.invoke(prompt)

print(completion)
