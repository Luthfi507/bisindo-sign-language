from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI()

async def speech(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions="Speak like Indonesian native speaker",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)