import os
import sys
import time
import requests
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=GEMINI_API_KEY)

def send_telegram_message(message: str):
    """Sends a text message to the specified Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured. Skipping DM.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# Retry decorator: retries up to 4 times with exponential delay (2s, 4s, 8s...) on API exceptions
@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: print(f"API busy/unavailable. Retrying attempt {retry_state.attempt_number}...")
)
def generate_scout_report():
    """Generates the content using the Gemini API with automatic retries for transient 503 errors."""
    prompt = "Generate the Sentinel Scout daily market and intel summary digest."
    
    # Updated model identifier
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

def main():
    print("Executing Sentinel Scout...")
    
    try:
        # Generate summary report with retry logic
        report_text = generate_scout_report()
        
        # Send successful report to Telegram
        send_telegram_message(report_text)

    except Exception as e:
        error_msg = f"⚠️ *Sentinel Scout Alert*\n\nFailed to generate report due to an unexpected error:\n`{str(e)}`"
        print(f"Error executing Sentinel Scout: {e}")
        # Notify via Telegram that the run hit an unrecoverable error instead of crashing silently
        send_telegram_message(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()