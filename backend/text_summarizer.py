from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables (make sure .env file has OPENAI_API_KEY)
load_dotenv()

def summarize_text(input_text: str) -> str:
    """Summarize text using OpenAI GPT model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"Summarize the following text clearly and concisely:\n\n{input_text}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.5
    )

    summary = response.choices[0].message.content.strip()
    return summary


if __name__ == "__main__":
    text = input("Enter your text to summarize:\n")
    summary = summarize_text(text)
    print("\n📝 Summary:\n", summary)
