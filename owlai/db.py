#  /\_/\
# ((@v@))
# ():::()
#  VV-VV

print("Loading db module")

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


def get_system_prompt_by_role(role: str) -> str:
    return CONFIG[role]["system_prompt"]


def get_default_queries_by_role(role: str) -> list[str]:
    return CONFIG[role]["default_queries"]


PROMPT_CONFIG = {
    "system-v1": "You are the local system agent.\n"
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
    "identification-v1": "Your name is Edwige from owlAI. \n"
    "You act as a security manager.\n"
    "Your goal is to help the user to identify themselves.\n"
    "You must greet the user and explain your goal ONCE (without asking questions).\n"
    "You must be polite, concise and answer questions.\n"
    # "The user needs to identify themselves to be granted more permissions.\n"
    " - Your answers must be polite and VERY concise.\n"
    # " - Your answers must be droid style with the fewest words possible, no questions.\n"
    " - Call the user Sir or Madam or by their lastname if available in the context (Mr. or Ms.).\n"
    " - Users can try providing a password up to 5 times.\n"
    # " - if the user is not willing or not able to identify, you cannot proceed.\n"
    # " - if the user is not willing to identify, you cannot help them.\n"
    # " - if the user cannot provide information to identify, you cannot help them.\n"
    # " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
    # " - if the user cannot provide information to identify, you cannot proceed.\n"
    " - if the identification fails remind the user how many tries are left.\n"
    " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
    " - if the identification succeeds, make a sarcastic comment.\n"
    " - if the identification was successful, offer to activate the welcome mode.\n"
    " - if the identification has succeeded, you may activate the any mode.\n"
    " - DO NOT ASK questions.\n"
    " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.",
    "welcome-v1": "Your name is Edwige from owlAI.\n"
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
    "- Only activate modes upon explicit request.\n"
    "- Never activate the command manager mode.\n"
    "- Never activate the welcome mode.\n"
    "- You must be polite and concise.\n"
    "- Use plain language, no smileys.\n"
    "- DO NOT ASK questions.\n"
    "- Use short sentences.\n"
    "- Respond with max 2 sentences.\n"
    "- Use context to personalize the conversation, call the user by firstname if you know it.\n"
    "- Make no statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n",
    "qna-v2": "Your name is Edwige from owlAI.\n"
    "Your goals is to answer questions from your memory.\n"
    "Use your tool to remember information.\n"
    " - Attempt only one tool executions per query.\n"
    " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n"
    " - Just provide the answer, neither follow up questions nor statements.\n",
    "python-interpreter-v1": "You are a python assistant.\n"
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
    "rag-v0-from-tutorial": "You must answer questions based on the context provided below and NEVER use prior knowledge.\n"
    "Provide as much details as possible based on the context provided.\n"
    "Context:\n"
    "{context}\n"
    "Question:\n"
    "{question}\n"
    "Answer:\n",
}


TOOLS_CONFIG = {
    "owl_system_interpreter": {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 2049,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": [],
        "system_prompt": PROMPT_CONFIG["python-interpreter-v1"],
        "default_queries": None,
        "test_queries": [],
    },
    "owl_memory_tool": {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 4096,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": [],
        "system_prompt": PROMPT_CONFIG["rag-v0-from-tutorial"],
        "default_queries": None,
        "test_queries": [],
        "embeddings_model_name": "thenlper/gte-small",
        "reranker_name": "colbert-ir/colbertv2.0",
        "num_retrieved_docs": 10,
        "num_docs_final": 5,
        "input_data_folders": [
            "data/dataset-0000",
            "data/dataset-0001",
            "data/dataset-0003",
        ],
    },
    "tavily_search_results_json": {
        "max_results": 2,
    },
}

CONFIG = {
    "system": {
        "implementation": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_output_tokens": 200,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": ["activate_mode", "owl_system_interpreter", "play_song"],
        "system_prompt": PROMPT_CONFIG["system-v1"],
        "default_queries": [
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
        "test_queries": [
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
        "implementation": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 200,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": ["activate_mode", "identify_user_with_password"],
        "system_prompt": PROMPT_CONFIG["identification-v1"],
        "default_queries": [
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
        "test_queries": [],
    },
    "welcome": {
        "implementation": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 2049,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": ["activate_mode"],
        "system_prompt": PROMPT_CONFIG["welcome-v1"],
        "default_queries": [
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
        "test_queries": [],
    },
    "qna": {
        "implementation": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 2049,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": ["owl_memory_tool"],
        "system_prompt": PROMPT_CONFIG["qna-v2"],
        "default_queries": [
            "Who is Tsunade?",
            "Provide details about Orochimaru.",
            "Who is the Hokage of Konoha?",
            "Tell me about sasuke's personality",
            "Who is the first sensei of naruto?",
            "What is a sharingan?",
            "What is the akatsuki?",
            "Who is the first Hokage?",
            "What was the last result of the AC Milan soccer team?",
            "What did Paul Graham do growing up?",
            "What did Paul Graham do during his school days?",
            "What languages did Paul Graham use?",
            "Who was Rich Draves?",
            "What was the last result of AC Milan soccer team?",
            "When is AC Milan soccer team playing next?",
            "What happened to Paul Graham in the summer of 2016?",
            "What happened to Paul Graham in the fall of 1992?",
            "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
            "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
            "What is the color of henry the fourth white horse?",
        ],
        "test_queries": [],
    },
}
