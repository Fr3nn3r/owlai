import pyttsx3
import threading
import pythoncom  # Import this to initialize COM in threads

engine = None

def speak(text):
    """Runs pyttsx3 in a separate thread. Couldn't get it to work otherwise"""
    def run():
        global engine
        pythoncom.CoInitialize()  # Initialize COM in this thread
        if engine is None:
            engine = pyttsx3.init()

            engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate+30)

        engine.say(text)
        engine.runAndWait()
        pythoncom.CoUninitialize()  # Clean up COM

    t = threading.Thread(target=run)
    t.start()
    return t  # Returning the thread if you need to track it

def _hoot(text : str):
    engine.say(text)
    engine.runAndWait()

def hoot_local(text : str):
    thread = speak(text)