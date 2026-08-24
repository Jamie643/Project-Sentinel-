import os
import datetime
import requests
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def fetch_github_trending():
    date_seven_days_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q=created:>{date_seven_days_ago}&sort=stars&order=desc&per_page=5"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        items = res.json().get("items", [])
        return [{"name": item["full_name"], "stars": item["stargazers_count"], "description": item["description"], "url": item["html_url"]} for item in items]
    except Exception as e:
        print(f"Error fetching GitHub: {e}")
        return []

def fetch_polymarket_events():
    url = "https://gamma-api.polymarket.com/events?limit=5&active=true&closed=false"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        events = res.json()
        return [{"title": e.get("title"), "volume": e.get("volume"), "description": e.get("description", "")[:150]} for e in events]
    except Exception as e:
        print(f"Error fetching Polymarket: {e}")
        return []

def generate_sentinel_briefing(github_data, polymarket_data):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    You are Project Sentinel's Scout Agent. Deliver ONE high-expected-value opportunity for an engineer/builder.
    
    GitHub Trending Data:
    {github_data}
    
    Polymarket Events Data:
    {polymarket_data}
    
    Output Format (under 150 words total, mobile friendly):
    🎯 **SENTINEL DAILY SCOUT**
    
    **Selected Opportunity:** [Name of project/market trend]
    **Signal Source:** [GitHub / Polymarket]
    
    **What Happened:** [Brief factual summary]
    **Why It Matters:** [Strategic angle / market gap]
    **Initial Action:** [Build / Learn / Invest / Ignore]
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("Telegram briefing delivered successfully.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def run_scout():
    print("Executing Sentinel Scout...")
    github_data = fetch_github_trending()
    polymarket_data = fetch_polymarket_events()
    
    if not github_data and not polymarket_data:
        print("No signals retrieved. Aborting.")
        return
        
    briefing = generate_sentinel_briefing(github_data, polymarket_data)
    if briefing:
        send_telegram_message(briefing)

if __name__ == "__main__":
    run_scout()
