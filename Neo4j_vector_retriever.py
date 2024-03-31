import yaml
import constants
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# Load OpenAI API key
CONFIG_FILE = constants.config

# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']

# Load the OpenAI API key
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define the prompt template with the input variables
prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History: {chat_history}
    Question: {input}
    """,
    input_variables=["chat_history", "input"]
)

# Define the memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the chat chain
chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Load the YouTube search tool
youtube = YouTubeSearchTool()

# Define the Neo4j vector retriever
movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=config['Neo4j']['url'],
    username=config['Neo4j']['username'],
    password=config['Neo4j']['password'],
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)

# Define the retrieval QA chain
plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True
)


def run_retriever(query):
    results = plot_retriever.invoke({"query":query})
    # format the results
    movies = '\n'.join([doc.metadata["title"] + " - " + doc.page_content for doc in results["source_documents"]])
    return movies

# Define the tools for the agent
tools = [
    # Define the movie chat tool
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True
    ),
    # Define the movie trailer search tool
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True
    ),
    Tool.from_function(
        # Define the movie plot search tool
        name="Movie Plot Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )
]

# Load the agent prompt
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_interations=3,
    verbose=True,
    handle_parse_errors=True)

while True:
    q = input("> ")
        # Check if the user wants to quit
    if q.lower() == "quit" or q.lower() == "exit":
        print("Exiting program...")
        break
    response = agent_executor.invoke({"input": q})
    print(response["output"])