from openai import OpenAI

client = OpenAI(
 base_url="http://localhost:11434/v1/",
)

response = client.chat.completions.create(
 messages=[
     {
         "role": "user",
         "content": "Say this is a test",
     }
 ],
 # notice the change in the model name
 model="llama3.2",
)

print(response.choices[0].message.content)