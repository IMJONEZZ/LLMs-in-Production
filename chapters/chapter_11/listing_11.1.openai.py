import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:8080/v1",  # replace with your pi's ip address
    api_key="1234",  # replace with your server's api key
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are Capybara, an AI assistant. Your top "
            "priority is achieving user fulfillment via helping them with "
            "their requests.",
        },
        {
            "role": "user",
            "content": "Building a website can be done in 10 simple steps:",
        },
    ],
)

print(completion.choices[0].message)
