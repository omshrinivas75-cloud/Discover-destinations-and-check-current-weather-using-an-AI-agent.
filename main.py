import os 
import certifi
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain_core.tools import tool
# ========================================
# LOAD ENV VARIABLES
# ========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
# ========================================
# LOAD ENV VARIABLES
# ========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEARCH_TOOL = TavilySearchResults(
    max_result = 2 
)
#  llm model 

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    api_key = GEMINI_API_KEY,
    temprature = 0
)
# frecth the prompt from langchain hub 
from langsmith import Client

client = Client()
prompt = repr(client.pull_prompt("hwchase17/react-multi-input-json",dangerously_pull_public_prompt=True))


import requests
from langchain_core.tools import tool

@tool
def Get_Weather(city: str) -> str:
    """Get the current weather information for a given city."""
    
    url = "https://api.weatherstack.com/current"
    
    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        return f"Weather API error: {data['error']['info']}"

    current = data["current"]
    location = data["location"]

    return (
        f"City: {location['name']}\n"
        f"Temperature: {current['temperature']}°C\n"
        f"Weather: {current['weather_descriptions'][0]}\n"
        f"Humidity: {current['humidity']}%"
    )

# ============================================
#     tools 
tools = [SEARCH_TOOL,Get_Weather]
#    Create Agent 

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = prompt
    
)
response = agent.invoke(
    {"messages":{"role":"user","content":"find best palce in indore for visit and also show the weather of city "}}
)