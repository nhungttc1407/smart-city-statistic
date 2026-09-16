# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="https://api.int2.net/v1",
    api_key="sk-Whkve2vfGhKsPB8bJBUJjw",
)

response = client.chat.completions.create(
    model="glm-5.3-flash",
    messages=[
        {"role": "user", "content": "Hello, who are you?"},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
