from RealtimeTTS import TextToAudioStream, BaseEngine, AzureEngine, ElevenlabsEngine
import os
import threading
import subprocess
import logging
import logging.config
import logging
import yaml

logger = logging.getLogger("ttsengine")

################################# TTS

tts_engine_name = "azure"
tts_engine = None
tts_stream = None

if tts_engine_name == "elevenlabs":
    tts_engine = ElevenlabsEngine(
        voice="Jane",
        #id="Xb7hH8MSUJpSbSDYk0k2",
        #id="RILOU7YmBhvwJGDGjNmP",
        id="bMxLr8fP6hzNRRi9nJxU",
        model="eleven_flash_v2",
        clarity=30,
        stability=30,
        #style_exxageration= 90,
)
elif tts_engine_name == "azure":
    tts_engine = AzureEngine(
        os.getenv("AZURE_SPEECH_KEY"),
        os.getenv("AZURE_SPEECH_REGION"),
        voice="en-US-AvaNeural",
        rate=30,
)
    tts_engine.emotion = "excited"
    tts_engine.emotion_role = "Girl"
elif tts_engine_name == "piper":
    logger.warning("Piper is not supported.")
    #tts_engine = PiperEngine()
elif tts_engine_name == "system" :
    logger.warning("System is not supported.")
    #tts_engine = SystemEngine()
elif tts_engine_name == "coqui" :
    logger.warning("Coqui is not supported.")
    #tts_engine = CoquiEngine()
    #tts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
else:
    raise ValueError(f"Unsuported TTS engine: {tts_engine_name}")   

tts_stream = TextToAudioStream(tts_engine)

def hoot(text : str):
    if tts_stream.is_playing() : tts_stream.stop()
    tts_stream.feed(text)
    tts_stream.play_async()

################################# END TTS