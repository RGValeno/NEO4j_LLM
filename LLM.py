import constants
import yaml
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

# Load OpenAI API key
CONFIG_FILE = constants.config

# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Create an instance of the OpenAI class with the specified API key, model, and temperature
llm = OpenAI(
    openai_api_key = OPENAI_API_KEY,
    model = constants.openAiModel,
    temperature = constants.temperature
)
# response = llm.invoke("What is a graph database?")

template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""")

# response = llm.invoke(template.format(fruit="apple"))

llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser=SimpleJsonOutputParser()
)

response = llm_chain.invoke({"fruit": "apple"})

print(response)