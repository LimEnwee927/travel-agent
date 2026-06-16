import requests
import os


SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
def search_web(query: str) -> str:
    """Search the web for travel information."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}
    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json().get("organic", [])
        output = []
        for r in results[:3]:
            output.append(f"- {r['title']}: {r['snippet']}")
        return "\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("cod") != 200:
            return f"Could not get weather for {city}."
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return f"{city}: {weather}, {temp}°C, humidity {humidity}%"
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"

def search_hotels(city: str, checkin: str = None, checkout: str = None) -> str:
    """Search for hotels using Serper web search."""
    query = f"best hotels in {city} 2024"
    if checkin:
        query += f" checkin {checkin}"
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}
    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json().get("organic", [])
        output = []
        for r in results[:3]:
            output.append(f"- {r['title']}: {r['snippet']}")
        return "\n".join(output) if output else "No hotels found."
    except Exception as e:
        return f"Hotel search failed: {str(e)}"

# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for travel attractions, restaurants, tips, or any travel-related information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query e.g. 'top attractions in Tokyo 2024'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city to help plan the trip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. 'Singapore'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. 'Bangkok'"
                    },
                    "checkin": {
                        "type": "string",
                        "description": "Check-in date e.g. '2024-12-01' (optional)"
                    },
                    "checkout": {
                        "type": "string",
                        "description": "Check-out date e.g. '2024-12-03' (optional)"
                    }
                },
                "required": ["city"]
            }
        }
    }
]