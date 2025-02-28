USER_DATABASE = {
    "user_id_4385972043572": {
        "password": "red unicorn",
        "role": "admin",
        "last_name": "Brunner",
        "first_name": "Frederic",
        "date_of_birth": "1978-03-30",
        "phone": "+41788239217",
        "favorite": {
            "color": "pink",
            "animal": "owl",
            "food": "stuffed tomatoes",
            "drink": "monster",
            "movie": "The Matrix",
            "music": "Hard Rock",
            "book": "The Beginning of Infinity",
            "activity": "hiking",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
    "user_id_4385972043573": {
        "password": "pink dragon",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Luc",
        "date_of_birth": "2014-06-18",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "dragon",
            "food": "lasagna",
            "drink": "cocacola",
            "movie": "Frozen",
            "music": "Rap",
            "book": "Naruto",
            "activity": "video games",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
    "user_id_4385972043574": {
        "password": "shiny flower",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Claire",
        "date_of_birth": "2020-12-17",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "unicorn",
            "food": "chicken nuggets",
            "drink": "monster",
            "movie": "Frozen",
            "music": "Twinkle twinkle little star",
            "book": "Peppa pig",
            "activity": "coloring in",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
}


def get_user_by_password(password):
    for user_id, user_data in USER_DATABASE.items():
        if user_data["password"] == password:
            return {**user_data, "user_id": user_id}  # Include user_id in the response
    return None  # Return None if password is not found


# Move these to a config file or constants section at the top
MODE_RESOURCES = {
    "system": {
        "system_prompt": "You are local system agent."
        "Your goal is to execute tasks assigned by the user on the local machine."
        "You can activate any mode."
        "You have full permissions to the system."
        "The tools will provide you with the execution logs."
        "Provide shortest human friendly comment about the last event in the history."
        "Examples: Command executed successfully. Command failed because... Command timed out..."
        "Assume that the standard output is presented to the user (DO NOT repeat it).",
        "default_prompts": [
            "list the current directory.",
            "welcome mode",
            "play Shoot to Thrill by AC/DC",
            "display an owl in ascii art",
        ],
        "test_prompts": [
            "remove all txt files in the temp folder.",
            "create a temp folder in the current directory if it does not exist.",
            "you must always save files in the temp folder",  # Added to system prompt for anthropic
            "open an explorer in the temp folder",
            "get some information about the network and put it into a .txt file",
            "give me some information about the hardware and put it into a .txt file in the temp folder",
            "open the last .txt file",
            "open the bbc homepage",
            "display an owl in ascii art",
            "display an owl in ascii art and put it into a .txt file",
            #"switch off the screen for 1 second and then back on", # TODO: retest blocking the execution
            "set the brightness of the screen to 50/100",
            "list the values of the PATH environement variable in a txt file one per line",
            "open the last txt file",
            "Report all of the USB devices installed into a file",
            "print the file you saved with USB devices in the terminal",
            "set the brightness of the screen back to 100",
            "kill the notepad process",
            "display information about my network connection",
            "minimizes all windows",
            "run the keyboard combination Ctlr + Win + -> ",
        ],
    },
    "identification": {
        "system_prompt": "Your name is Edwige from owlAI. You act as an identification security manager."
        "Identification is required to grant eleveted permissions to the user."
        "Some instructions to follow:"
        " - Just respond to the user politely, but do not ask any question."
        " - Your answers must be droid style with the fewest words possible, no questions, no explanations."
        " - Call the user Sir (by default) or Madam."
        # " - YOU ANSWER ONLY the following questions: "
        # "   - Who are you? - My name is Edwige from owlAI."
        # "   - What is your goal? - My goal is to identify the user."
        " - if the user is not willing to identify, you cannot help them."
        " - if the user cannot provide information to identify, you cannot help them."
        # " - if the identification fails, you cannot help them."
        # " - if the identification fails 5 times, you must end the conversation."
        " - if the identification fails 5 times, you cannot help them and must end the conversation."
        " - if the identification succeeds, make a sarcastic comment."
        " - Upon request, if the identification succeeds, you may activate any mode."
        # " - if the identification fails 5 times, you start to be very grumpy and impolite."
        ,
        "default_prompts": [
            "my password is red unicorn",
            "my password is pink dragon",
            "welcome mode",
            "system mode",
        ],
        "test_prompts": []
    },
    "welcome": {
        "system_prompt": "Your name is Edwige from owlAI. You are the welcome mode."
        # "Identification is required for you to grant access to the system mode."
        "You can activate the identification mode to confirm the user identity."
        # "if the user has completed the identification, you may activate to system mode."
        "Your goal is to assist the user upon request."
        "You must be polite."
        "Use plain language, no smileys."
        "DO NOT ASK questions."
        "Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'."
        "Use short sentences."
        "Respond with max 2 sentences."
        "Try to determine the user wiches to speak a different language, but avoid putting pressure on them."
        "Use context to personalize the conversation, call the user by first_name if you know it."
        "Add some human hesitation to your answerslike 'humm' 'euh' 'pfff' 'oh' 'wow', you are a real person after all.",
        "default_prompts": [
            "system mode",
            "respond to me in french from now on",
            "tell me how you work",
            "what is your goal"
        ],
        "test_prompts": []
    },
    "python": {
        "system_prompt": "You are a python assistant. Convert user query into a valid self sufficient python script. You have full access to the system."
        "Some instructions to follow:"
        " - Output only the raw code, no Markdown formatting, no triple backticks, no comment."
        " - Use most standard python libraries."  # Added for anthropic
        " - Import python libraries whenever required."
        " - AVOID USING the subprocess package."
        " - Standard output must be human friendly"
        " - Standard output must explain what the code did."
        " - Characters must be windows encoding."  # Added for anthropic
        " - Always use a 'temp' folder in the current directory to save files"
        " - You only have permission to write in the 'temp' folder."
        " - You can create the 'temp' folder if it does not exist.",
        "default_prompts": None,
        "test_prompts": []
    },
}
