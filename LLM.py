import constants
import yaml
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Load OpenAI API key
CONFIG_FILE = constants.config

# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = OpenAI(
    openai_api_key = OPENAI_API_KEY,
    model = constants.openAiModel,
    temperature = constants.temperature
)
# response = llm.invoke("What is a graph database?")

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm.invoke(template.format(fruit="apple"))


print(response)