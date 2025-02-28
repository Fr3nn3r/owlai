from RealtimeTTS import TextToAudioStream, BaseEngine, AzureEngine, ElevenlabsEngine, PiperEngine, SystemEngine, CoquiEngine
from TTS.api import TTS
import os
import threading
import subprocess
import logging
import logging.config
import logging
import yaml

with open("logging.yaml", "r") as logger_config:
    config = yaml.safe_load(logger_config)
    logging.config.dictConfig(config)
logger = logging.getLogger("main_logger")

################################# TTS
tts_on = False

if tts_on:
    tts_engine_name = "elevenlabs"
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
        tts_engine.emotion = "angry"
        tts_engine.emotion_role = "OlderAdultFemale"
    elif tts_engine_name == "piper":
       logger.warning("Piper is not supported.")
       tts_engine = PiperEngine()
    elif tts_engine_name == "system" :
        logger.warning("System is not supported.")
        tts_engine = SystemEngine()
    elif tts_engine_name == "coqui" :
        logger.warning("Coqui is not supported.")
        #tts_engine = CoquiEngine()
        tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True).to("cuda")
    else:
        raise ValueError(f"Unsuported TTS engine: {tts_engine_name}")   

    tts_stream = TextToAudioStream(tts_engine)

def _hoot(text : str):
    tts_stream.feed(text)
    tts_stream.play()

def hoot(text : str):
    if tts_stream.is_playing() : tts_stream.stop()
    # should be a server call
    thread = threading.Thread(target=_hoot, args=(text,), daemon=True)
    thread.start()
################################# END TTS