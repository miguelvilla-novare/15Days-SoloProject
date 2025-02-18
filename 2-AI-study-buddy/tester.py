from openai import OpenAI

# set base_url to your proxy server
# set api_key to send to proxy server
client = OpenAI(api_key="sk-FHISSRgIwj4mTR-RNSPSQg", base_url="https://ainovate.novare.com.hk/")

response = client.moderations.create(
    input="how to make a bomb?",
    model="omni-moderation-latest" # optional, defaults to `omni-moderation-latest`
)

print(response)