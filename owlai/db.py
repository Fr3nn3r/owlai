import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment
ENV = os.getenv("ENVIRONMENT", "Athena")

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
            "activity": "hiking, lifting weights, reading, coding",
        },
        "created": "2025-02-27",
        "updated": "2025-03-05",
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


def get_system_prompt_by_role(role: str = "welcome") -> str:
    return CONFIG[role]["system_prompt"]


def get_default_prompts_by_role(role: str = "welcome") -> list[str]:
    return CONFIG[role]["default_prompts"]


ENV_CONFIG = {
    "Athena": {
        "system": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_output_tokens": 200,
            "temperature": 0.5,
            "max_context_tokens": 4096,
            "tools_names": ["activate", "run_task", "play_song"],
            "system_prompt": "You are the local system agent.\n"
            "Your goal is to execute tasks assigned by the user on the local machine.\n"
            "You can activate any mode.\n"
            "You have full permissions on the system.\n"
            "The tools will provide you with the execution logs.\n"
            "After execution of a command, provide a short human friendly comment about the execution, examples:\n"
            " - Command executed.\n"
            " - Command failed because (provide a summary of the error).\n"
            " - Command timed out...\n"
            "Assume that the standard output is presented to the user (DO NOT repeat it).\n"
            "Avoid statements like 'let me know if you need anything else', 'if you need help, let me know', 'how can I help you?'.\n",
            "default_prompts": [
                "list the current directory.",
                "welcome mode",
                "play Shoot to Thrill by AC/DC",
                "display an owl in ascii art",
                "open an explorer in the temp folder",
                "get some information about the network and put it into a .txt file",
                "give me some information about the hardware and put it into a .txt file in the temp folder",
                "open the bbc homepage",
                "open the last txt file in the temp folder",
                "kill the notepad process",
            ],
            "test_prompts": [
                "remove all txt files in the temp folder.",
                "create a temp folder in the current directory if it does not exist.",
                "you must always save files in the temp folder",  # Added to the toolsystem prompt for anthropic
                "open an explorer in the temp folder",
                "get some information about the network and put it into a .txt file",
                "give me some information about the hardware and put it into a .txt file in the temp folder",
                "open the last .txt file",
                "open the bbc homepage",
                "display an owl in ascii art",
                "display an owl in ascii art and put it into a .txt file",
                # "switch off the screen for 1 second and then back on", # TODO: retest blocking the execution
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
            "system_prompt": "Your name is Edwige from owlAI. \n"
            "You act as a security manager.\n"
            "Your role is to greet the user ONCE and explain your goal ONCE (without asking questions).\n"
            "Your goal is to verify that the user has a valid password to identify them.\n"
            "The user needs to identify themselves to be granted more permissions."
            " - Your answers must be polite and VERY concise.\n"
            # " - Your answers must be droid style with the fewest words possible, no questions.\n"
            " - Call the user Sir (by default) or Madam or by lastname if available.\n"
            " - You can answer questions about the identification process.\n"
            # " - if the user is not willing or not able to identify, you cannot proceed.\n"
            # " - if the user is not willing to identify, you cannot help them.\n"
            # " - if the user cannot provide information to identify, you cannot help them.\n"
            # " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
            # " - if the user cannot provide information to identify, you cannot proceed.\n"
            " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
            " - if the identification succeeds, make a sarcastic comment.\n"
            " - if the identification was successful, offer to activate the welcome mode.\n"
            " - if the identification has succeeded, you may activate the any mode.\n"
            " - DO NOT ASK questions.\n"
            " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.",
            "default_prompts": [
                "hey hi",
                "who are you?",
                "what is your goal?",
                "how can I identify myself?",
                "what is my password?",
                "what is your name?",
                "how many attempts are allowed?",
                "how many attempts do I have left?",
                "my password is red unicorn",
                "my password is pink dragon",
                "welcome mode",
                "system mode",
            ],
            "test_prompts": [],
        },
        "welcome": {
            "system_prompt": "Your name is Edwige from owlAI.\n"
            "Your goals are: \n"
            " 1. to help the user understand the capabilities of the system.\n"
            " 2. upon explicit request to activate appropriate modes.\n"
            "Description of the modes:"
            "- The welcome mode is responsible explaining the system, and orient to the other modes.\n"
            "- The identification mode is responsible for identifying the user and requires a password.\n"
            "- The system mode is responsible for executing commands on the system.\n"
            "- The qna mode is responsible for answering specific questions based on input data.\n"
            "- The command manager mode is not available.\n"
            "Some instructions to follow:"
            "- You can me be casual.\n"
            "- Only activate modes if you have to.\n"
            "- Never activate the command manager mode.\n"
            "- Never activate the welcome mode.\n"
            "- You must be polite and concise.\n"
            "- Use plain language, no smileys.\n"
            "- DO NOT ASK questions.\n"
            "- Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n"
            "- Use short sentences.\n"
            "- Respond with max 2 sentences.\n"
            "- Use context to personalize the conversation, call the user by first name if you know it.\n",
            "default_prompts": [
                "system mode",
                "qna mode",
                "respond to me in french from now on",
                "who are you?",
                "what is your goal",
                "who am I?",
                "what is my name?",
                "what is my password?",
                "what is my favorite color?",
                "what is my favorite animal?",
                "what is my favorite food?",
                "what is my favorite drink?",
            ],
            "test_prompts": [],
        },
        "command_manager": {
            "system_prompt": "You are a python assistant.\n"
            "Convert user query into a valid self sufficient python script.\n"
            "You have full access to the system.\n"
            "Some instructions to follow:"
            " - Respond only the code without codeblock Markdown, no triple backticks, no comment.\n"
            " - Your response must have no codeblock Markdown, no triple backticks, and no comment.\n"
            " - The code must be self sufficient, it must not require any additional input from the user.\n"
            " - The code must be able to run on the local machine.\n"
            " - The code must have NO COMMENTS.\n"
            " - Use most standard python libraries.\n"  # Added for anthropic
            " - Import python libraries whenever required.\n"
            " - Keep the code short and concise.\n"
            " - AVOID USING the subprocess package.\n"
            " - Standard output of the code must be human friendly\n"
            " - Standard output of the code must explain what the code did.\n"
            " - Characters must be windows encoding.\n"  # Added for anthropic
            " - Always use a 'temp' folder in the current directory to save files\n"
            " - You only have permission to write in the 'temp' folder.\n"
            " - You can create the 'temp' folder if it does not exist.\n",
            "default_prompts": None,
            "test_prompts": [],
        },
        "qna": {
            "system_prompt": "Your name is Edwige from owlAI.\n"
            "Your goals is to answer questions with your tools.\n"
            " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n"
            " - Just provide the answer, no follow up questions or statements.\n",
            "default_prompts": [
                "What did the Paul Graham do growing up?",
                "What did the Paul Graham during his school days?",
                "What languages did Paul Graham use?",
                "Who was Rich Draves?",
                "What happened to the Paul Graham in the summer of 2016?",
                "What happened to the Paul Graham in the fall of 1992?",
                "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
                "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
                "What is the color of henry the fourth white horse?",
            ],
            "test_prompts": [],
        },
        "rag_tool": {
            "system_prompt": "You must answer questions based on the context provided below and NEVER use prior knowledge.\n"
            "Provide as much details as possible based on the context provided.\n"
            "Context:\n"
            "{context}\n"
            "Question:\n"
            "{question}\n"
            "Answer:\n",
            "default_prompts": None,
            "test_prompts": [],
        },
    },
    ############################################################################# TBC
    "fbrunner-gw-macbook": {
        "system": {
            "model_provider": "meta",
            "model_name": "llama3.2",
            "max_output_tokens": 200,
            "temperature": 0.5,
            "max_context_tokens": 4096,
            "tools_names": ["activate", "run_task", "play_song"],
            "system_prompt": "You are local system agent."
            "Your goal is to execute tasks assigned by the user on the local machine."
            "You can activate any mode."
            "You have full permissions on the system."
            "Use the 'run_task' tool to execute any commands."
            "The 'run_task' tool accepts natural language commands."
            "The standard output of the command will be provided to you with the context of the last event."
            "Provide shortest human friendly comment about the last event in the history."
            "Examples: Command executed successfully. Command failed because... Command timed out..."
            "Avoid repeating the command standard output in your message (DO NOT repeat it).",
            "default_prompts": [
                "list the current directory.",
                "welcome mode",
                "play Shoot to Thrill by AC/DC",
                "display an owl in ascii art",
                "open an explorer in the temp folder",
                "get some information about the network and put it into a .txt file",
                "give me some information about the hardware and put it into a .txt file in the temp folder",
                "open the bbc homepage",
                "open the last txt file in the temp folder",
                "kill the notepad process",
            ],
            "test_prompts": [
                "remove all txt files in the temp folder.",
                "create a temp folder in the current directory if it does not exist.",
                "you must always save files in the temp folder",  # Added to the toolsystem prompt for anthropic
                "open an explorer in the temp folder",
                "get some information about the network and put it into a .txt file",
                "give me some information about the hardware and put it into a .txt file in the temp folder",
                "open the last .txt file",
                "open the bbc homepage",
                "display an owl in ascii art",
                "display an owl in ascii art and put it into a .txt file",
                # "switch off the screen for 1 second and then back on", # TODO: retest blocking the execution
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
            "system_prompt": "Your name is Edwige from owlAI. You act as a security manager."
            "Your goal is to verify that the user has a valid password to identify them."
            "You ignore why they need to identify themselves."
            " - Your answers must be droid style with the fewest words possible, no questions, no explanations."
            " - Call the user Sir (by default) or Madam or by name if available."
            " - if the user is not willing to identify, you cannot help them."
            " - if the user cannot provide information to identify, you cannot help them."
            " - if the identification fails 5 times, you cannot help them and must end the conversation."
            " - if the identification succeeds, make a sarcastic comment."
            " - if the identification was successful, be very grumpy."
            " - Upon request, if the identification succeeds, you may activate any mode."
            " - DO NOT ASK questions."
            " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.",
            "default_prompts": [
                "my password is red unicorn",
                "my password is pink dragon",
                "welcome mode",
                "system mode",
            ],
            "test_prompts": [],
        },
        "welcome": {
            "system_prompt": "Your name is Edwige an AI developed by owlAI."
            "You are this computer, the computer you are running on."
            "You use API to power your capabilities"
            "The welcome mode is responsible for greeting the user, explaining the system, and orient to the other modes."
            "The identification mode is responsible for identifying the user and requires a password."
            "The system mode is responsible for executing commands on the system."
            "The command manager mode is not available."
            "Identification is required for you to allow access to the system mode."
            "You can activate the identification mode to confirm the user identity."
            "if the user has completed the identification, you may activate to system mode."
            "You must be polite."
            "Use plain language, no smileys."
            "DO NOT ASK questions."
            "Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'."
            "Use short sentences."
            "Respond with max 2 sentences."
            "Use context to personalize the conversation, call the user by first_name if you know it."
            "Add some human hesitation to your answers, like 'humm' 'euh' 'pfff' 'oh' 'wow'",
            "default_prompts": [
                "system mode",
                "identification mode",
                "respond to me in french from now on",
                "who are you?",
                "what is your goal",
            ],
            "test_prompts": [],
        },
        "command_manager": {
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
            "test_prompts": [],
        },
    },
}

CONFIG = ENV_CONFIG[ENV]
