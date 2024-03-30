import yaml
from langchain_openai import OpenAI
import constants
# Load OpenAI API key
OpenAi_CONFIG_FILE = constants.config

# Load the config file
with open(OpenAi_CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# print(type(config))

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


llm = OpenAI(openai_api_key=OPENAI_API_KEY)

response = llm.invoke("What is Neo4j?")

print(response)