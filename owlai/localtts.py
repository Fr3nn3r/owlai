import pyttsx3
import threading

engine = pyttsx3.init()

voices = engine.getProperty('voices')
english_female = voices[2]
french_female = voices[3]
italian_female = voices[4]
german_female = voices[1]


engine.setProperty('voice', voices[2].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate+30)

def _hoot(text : str):
    engine.say(text)
    engine.runAndWait()

def hoot_local(text : str):
    engine.stop()
    # should be a server call
    thread = threading.Thread(target=_hoot, args=(text,), daemon=True)
    thread.start()