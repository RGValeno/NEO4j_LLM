import constants
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

# Load OpenAI API key
CONFIG_FILE = constants.config

# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']

chat_llm = ChatOpenAI(
    openai_api_key = OPENAI_API_KEY
)

prompt = PromptTemplate(
    template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)

prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Chat History: {chat_history}
Context: {context}
Question: {question}
""", input_variables=["chat_history", "context", "question"])


chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory, verbose=True)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

# response = chat_chain.invoke(
#     {
#         "context": current_weather,
#         "question": "What is the weather like on Watergate Bay?",
#     }
# )

while True:
    question = input("> ")
    response = chat_chain.invoke({
        "context": current_weather,
        "question": question
        })

    print(response["text"])

print(response)