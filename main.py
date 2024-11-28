import asyncio
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, cartesia 


class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""
    
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Translate user-provided text or numbers into the desired language in a simple, easy-to-understand style. Stick strictly to the requested language and person-first language, avoiding explanations, punctuation, or extra details. Maintain the original grammatical perspective (first person stays first person)"

                ),
            )
        ]
    )

    llava = openai.LLM.with_groq(model="llama-3.1-70b-versatile")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=cartesia.TTS(model="sonic-multilingual",language="hi",voice="c1abd502-9231-4558-a054-10ac950c356d"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer()
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),  #  use Deepgram's Speech To Text (STT)
        llm=llava,
        tts=openai_tts,  # use Cartesia's TTS
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Translate what the user said into the desired language.
        """


        chat_context.messages.append(ChatMessage(role="user", content=text))

        stream = llava.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Please say the language to translate", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))