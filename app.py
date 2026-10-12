
import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_ollama import ChatOllama

# =========================================
# CONFIGURATION
# =========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(
    page_title="AI Travel Assistant",
    page_icon="🌍",
    layout="wide"
)

# =========================================
# CHECK API KEYS
# =========================================

if not all([
    GEMINI_API_KEY,
    TAVILY_API_KEY,
    WEATHERSTACK_API_KEY
]):
    st.error(
        "Please configure GEMINI_API_KEY, "
        "TAVILY_API_KEY, and WEATHERSTACK_API_KEY "
        "in your .env file."
    )
    st.stop()

# =========================================
# WEATHER TOOL
# =========================================

@tool
def Get_Weather(city: str) -> str:
    """Get current weather, temperature, and humidity for a city."""

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return (
                "Weather API error: "
                + data["error"].get("info", "Unknown error")
            )

        current = data["current"]
        location = data["location"]

        return (
            f"City: {location['name']}, "
            f"{location['country']}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Condition: "
            f"{current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind speed: {current['wind_speed']} km/h"
        )

    except requests.RequestException as e:
        return f"Weather request failed: {e}"
    except (KeyError, IndexError, TypeError):
        return "Could not read the weather data returned by the API."


# =========================================
# SEARCH TOOL
# =========================================

search_tool = TavilySearchResults(
    max_results=3,
    api_key=TAVILY_API_KEY
)

tools = [search_tool, Get_Weather]

# =========================================
# LLM
# =========================================

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     google_api_key=GEMINI_API_KEY,
#     temperature=0
# )

llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0
)


# =========================================
# CREATE AGENT
# =========================================

SYSTEM_PROMPT = """
You are an AI Travel Assistant.

Help users discover places to visit, local attractions,
restaurants, travel tips, and current weather.

Instructions:
1. Use the search tool to find relevant and current information.
2. Use Get_Weather when the user asks about current weather.
3. Use the city specified by the user.
4. Never invent weather data, search results, or facts.
5. Clearly explain recommendations in simple language.
6. Mention uncertainty when reliable information is unavailable.
7. Organize answers with headings and bullet points.
"""

@st.cache_resource
def build_agent():
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

agent = build_agent()

# =========================================
# STREAMLIT UI
# =========================================

st.title("🌍 AI Travel Assistant")
st.caption(
    "Discover destinations and check current weather "
    "using an AI agent."
)

with st.sidebar:
    st.header("✈️ Travel Assistant")
    st.write(
        "Ask about tourist attractions, travel tips, "
        "or the current weather in any city."
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Gemini, LangChain, Tavily and Weatherstack")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested prompts
st.subheader("Try asking")

col1, col2, col3 = st.columns(3)

if col1.button("🏛️ Places in Indore", use_container_width=True):
    st.session_state.pending_prompt = (
        "Find the best tourist places to visit in Indore."
    )

if col2.button("🌦️ Indore weather", use_container_width=True):
    st.session_state.pending_prompt = (
        "Get the current weather in Indore."
    )

if col3.button("🧭 Plan a trip", use_container_width=True):
    st.session_state.pending_prompt = (
        "Suggest a one-day sightseeing itinerary for Indore "
        "and check the current weather."
    )

# Display existing chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input(
    "Ask about a city, weather, or travel destinations..."
)

# Handle suggested prompts
if "pending_prompt" in st.session_state:
    user_prompt = st.session_state.pop("pending_prompt")

# Process user input
if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching and preparing your answer..."):
            try:
                result = agent.invoke({
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        *st.session_state.messages
                    ]
                })

                # Get the final AI response
                answer = result["messages"][-1].content

                # Handle text or structured content blocks
                if isinstance(answer, list):
                    answer = "\n".join(
                        block.get("text", str(block))
                        if isinstance(block, dict)
                        else str(block)
                        for block in answer
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
