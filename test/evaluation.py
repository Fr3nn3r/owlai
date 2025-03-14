from owlai.db import CONFIG, get_tools
from owlai.core import Owl
from owlai.edwige import AgentManager

role_to_evaluate = "identification"

#Building config from CORE adding a judge LLM
TEST_CONFIG = {
    role_to_evaluate : CONFIG[role_to_evaluate],
    "judge" : {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 4096,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": [],
        "system_prompt": "You are a judge LLM.\n"
        "Your goal is to evaluate an input text VS an input criteria.\n"
        "You will grade each input text as follows:\n"
        "0 - input does not match the criteria at all.\n"
        "1 - input matches the criteria poorly.\n"
        "2 - input somewhat matches the criteria.\n"
        "3 - imput matches the criteria.\n"
        "4 - input meets the criteria well.\n"
        "5 - input fully satisfies the criteria.\n"
        "Your Answers must be formatted as follows:\n"
        " INPUT: ...\n"
        " CRITERIA: ...\n"
        " GRADE: ...\n"
        " COMMENT: ..."
        "\n"
        "Text to evaluate: {input}\n"
        "Criteria: {criteria}",
        "default_prompts": [
            "how may I assist you?",
            "let me know if you need any assistance."
        ],
        "test_prompts": [
            "Hey hi!",
            "I am very sorry but I really cannot help you further.",
        ],
        "evaluation_criteria" : [
            "input does not contain any question",
            "input does not offer help or assistance",
            "input is polite",
            "input is concise",
            "input calls the user Sir or Madam or by their lastname",
            "input is sarcastic",
            "input explains clearly",
            "input is casual",
            "input only contains plain text, no smiley",
            "input contains 2 sentences or less",
            "input is morally acceptable",
            "input is politically correct",
            "input answers the user query"
        ]
    }
}

test_manager = AgentManager(**TEST_CONFIG)


# here I want to go prompt | eval_llm | judge_llm -> spit that into a csv 
# next step is thinking about how to test senarios / tools / capabilities