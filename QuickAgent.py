import asyncio
import shutil
import subprocess
import requests
import time
import os
import re
import ssl
import io
from openai import OpenAI

context = ssl.create_default_context()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key="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")
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key="sk-proj-uevHpVqJ6bfhnXfxTqALT3BlbkFJLe7kPMjGy3sq0SRlk0Bm")
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key="sk-proj-uevHpVqJ6bfhnXfxTqALT3BlbkFJLe7kPMjGy3sq0SRlk0Bm")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def clean_response(self, text):
        cleaned_text = re.sub(r'(<s>|</s>|`|"\[.*?\]"|"\[.*?\]")', '', text)
        return cleaned_text.strip()

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()
        response = self.conversation.invoke({"text": text})
        execution_time = time.time() - start_time
        cleaned_response = self.clean_response(response['text'])
        self.memory.chat_memory.add_ai_message(cleaned_response)
        print(execution_time)
        return cleaned_response

class TextToSpeechOpenAI:
    def __init__(self):
        self.client = OpenAI(api_key="sk-proj-uevHpVqJ6bfhnXfxTqALT3BlbkFJLe7kPMjGy3sq0SRlk0Bm")

    def speak(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )

        # Guardar el audio como un archivo temporal
        with open("temp_audio.mp3", "wb") as audio_file:
            audio_file.write(response.content)

        # Reproducir el audio utilizando ffplay.exe
        player_command = ["ffplay.exe", "-nodisp", "-autoexit", "temp_audio.mp3"]
        subprocess.run(player_command)

class TranscriptCollector:
    def __init__(self):
        self.transcript_parts = []

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()
    config = DeepgramClientOptions(options={"keepalive": "true"})
    deepgram = DeepgramClient("42da42105cb7ca70713a56ab7e846f4868af5653", config)
    dg_connection = deepgram.listen.asynclive.v("1")
    print("Listening...")

    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if not result.speech_final:
            transcript_collector.add_part(sentence)
        else:
            transcript_collector.add_part(sentence)
            full_sentence = transcript_collector.get_full_transcript().strip()
            if full_sentence:
                print(f"Human: {full_sentence}")
                callback(full_sentence)
                transcript_collector.reset()
                transcription_complete.set()

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(
        model="nova-2",
        punctuate=True,
        language="es-419",
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        endpointing=300,
        smart_format=True,
    )

    await dg_connection.start(options)
    microphone = Microphone(dg_connection.send)
    microphone.start()
    await transcription_complete.wait()
    microphone.finish()
    await dg_connection.finish()

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeechOpenAI()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            await get_transcript(handle_full_sentence)
            if "Nos vemos." in self.transcription_response.lower():
                break
            llm_response = self.llm.process(self.transcription_response)
            self.tts.speak(llm_response)
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())