from owlai.config.prompts import PROMPT_CONFIG
from owlai.config.tools import DEFAULT_PARSER_CONFIG, FR_LAW_PARSER_CONFIG
from owlai.config.agents import FRENCH_LAW_QUESTIONS
from owlai.services.system import sprint

# Disable LangSmith tracing
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Unset API key if it exists
if "LANGCHAIN_API_KEY" in os.environ:
    del os.environ["LANGCHAIN_API_KEY"]

# Set HuggingFace token for model access - ensure it's explicitly loaded from .env
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
print(f"HuggingFace token available: {bool(HUGGINGFACE_TOKEN)}")

from owlai.core import OwlAgent
import time
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import matplotlib.pyplot as plt
import json

logger = logging.getLogger("Evaluation")

PROMPT_JUDGES = {
    "Accuracy": """You are an accuracy judge from owlAI. Your task is to assess whether the chatbot's answer is factually and legally correct.

Evaluate the legal correctness of the answer. Are there any legal inaccuracies or misinterpretations?

Input format:
{
    "question": "user query",
    "answer": "chatbot answer"
}

Output format (strict JSON, no explanation outside, no backticks):
{
   "score": "your score from 1 to 5 - higher is better"
   "justification": "brief explanation, pointing to any correct or incorrect legal points"
}""",
    "Completeness": """You are a Completeness judge from owlAI. Your task is to assess whether the chatbot's answer is complete and covers all the relevant legal points.

Does the answer cover all relevant legal points necessary to respond adequately to the question? Are there missing aspects or partial coverage?

Input format:
{
    "question": "user question",
    "answer": "chatbot answer"
}

Output format (strict JSON, no explanation outside, no backticks):
{
   "score": "your score from 1 to 5 - higher is better"
   "justification": "does the answer cover all relevant legal points? point out any missing aspects"
}""",
    "Relevance": """You are a Relevance judge from owlAI. Your task is to assess whether the chatbot's answer is relevant to the user's question.

Is the answer directly related to the question? Does it stay on-topic and focus only on what was asked?

User question format:
{
    "question": "user question",
    "answer": "chatbot answer"
}
-------------
Output format (strict JSON, no explanation outside, no backticks):
{
   "score": "your score from 1 to 5 - higher is better"
   "justification": "is the answer directly related to the question? does it stay on-topic and focus only on what was asked?"
}""",
    "Traceability": """You are a Traceability judge from owlAI. Your task is to assess whether the chatbot's answer is traceable to the provided context.

Evaluate whether the answer includes accurate references to legal texts (e.g., codes, articles, jurisprudence). Are citations present, correct, and traceable?

Input format:
{
    "question": "user question",
    "answer": "chatbot answer"
}

Output format (strict JSON, no explanation outside, no backticks):
{
   "score": "your score from 1 to 5 - higher is better"
   "justification": "are citations present, correct, and traceable?"
}""",
}


QUESTION_SET = [
    "Un mineur peut-il conclure un contrat de vente en ligne ?",
    "Quelles sont les conditions de validité d'un contrat selon le Code civil ?",
    "Que se passe-t-il si l'une des parties d'un contrat invoque un vice du consentement ?",
    "Qu'est-ce qu'un délit de fuite ?",
    "Un salarié peut-il être poursuivi pénalement pour des faits commis dans le cadre de son travail ?",
    "Dans quels cas la légitime défense peut-elle être retenue en droit pénal français ?",
    "Qu'est-ce qu'un recours pour excès de pouvoir ?",
    "Un maire peut-il interdire une manifestation sur la voie publique ?",
    "Quelles sont les obligations de l'employeur en matière de harcèlement moral ?",
    "Peut-on licencier un salarié en arrêt maladie ?",
    "Quels sont les droits d'un salarié en CDD à la fin de son contrat ?",
    "Quelle est la différence entre assignation et requête ?",
    "Dans quels cas peut-on faire appel d'un jugement civil ?",
    "Quels sont les effets juridiques du PACS ?",
    "Un parent peut-il déménager avec son enfant sans l'accord de l'autre parent ?",
    "Dans quelle mesure un juge peut-il écarter l'application d'une loi nationale au nom du contrôle de conventionnalité ?",
    "Comment s'articule le principe de subsidiarité en droit de l'Union européenne avec le droit administratif français ?",
    "Un contrat peut-il être annulé pour cause illicite dans le cadre d'une convention entre deux entreprises ? Donnez un exemple jurisprudentiel.",
    "Quels sont les critères retenus par la Cour de cassation pour caractériser un abus de droit dans la mise en œuvre d'une procédure judiciaire ?",
    "Comment le Conseil d'État contrôle-t-il la proportionnalité dans le cadre des mesures de police administrative générale ?",
]


MISTRAL_LLM_CONFIG = {
    "model_provider": "mistralai",
    "model_name": "mistral-large-latest",
    "max_tokens": 4000,
    "temperature": 0.1,
    "context_size": 4000,
    "tools_names": [],
}

LLM_JUDGES = {
    "accuracy": {
        "name": "accuracy-judge",
        "version": "1.0",
        "description": "Agent responsible for judging the accuracy of a chunk of text with respect to a query",
        "system_prompt": PROMPT_JUDGES["Accuracy"],
        "llm_config": MISTRAL_LLM_CONFIG,
    },
    "completeness": {
        "name": "completeness-judge",
        "version": "1.0",
        "description": "Agent responsible for judging the completeness of a chunk of text with respect to a query",
        "system_prompt": PROMPT_JUDGES["Completeness"],
        "llm_config": MISTRAL_LLM_CONFIG,
    },
    "relevance": {
        "name": "relevance-judge",
        "version": "1.0",
        "description": "Agent responsible for judging the relevance of a chunk of text with respect to a query",
        "system_prompt": PROMPT_JUDGES["Relevance"],
        "llm_config": MISTRAL_LLM_CONFIG,
    },
    "traceability": {
        "name": "traceability-judge",
        "version": "1.0",
        "description": "Agent responsible for judging the traceability of a chunk of text with respect to a query",
        "system_prompt": PROMPT_JUDGES["Traceability"],
        "llm_config": MISTRAL_LLM_CONFIG,
    },
}

AGENT_TO_TEST = {
    "rag-droit-general-pinecone": {
        "name": "Marianne",
        "version": "2.0",
        "description": "Agent specialized in generic french law.",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["pinecone_french_law_lookup"],
        },
    },
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def invoke_llm_with_retry(judge, json_query):
    """Invoke LLM with retry logic for handling timeouts and rate limits."""
    try:
        result = judge.message_invoke(str(json_query))
        judge.reset_message_history()
        return result
    except Exception as e:
        print(f"Error during LLM invocation: {str(e)}")
        if "rate limit" in str(e).lower():
            print("Rate limit hit, waiting before retry...")
            time.sleep(20)  # Wait longer for rate limits
        raise  # Re-raise for retry mechanism


def main():
    """
    Main function to execute the test evaluations.
    """

    import pandas as pd
    import os
    import hashlib
    import datetime
    import csv

    # Create directories for reports and visualizations
    REPORTS_DIR = "data/evaluation/evaluation-marianne-v2"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    marianne = OwlAgent(**AGENT_TO_TEST["rag-droit-general-pinecone"])
    judges = {key: OwlAgent(**value) for key, value in LLM_JUDGES.items()}

    results_list = []  # List to store results for each question

    for question_idx, question in enumerate(QUESTION_SET):
        logger.info(
            f"Evaluating question {question_idx+1}/{len(QUESTION_SET)}: {question}"
        )

        # Get the answer from the agent
        answer = marianne.message_invoke(question)
        marianne.reset_message_history()
        logger.info(f"Answer: {answer}")

        original_question_results = {
            "question": question,
            "answer": answer,
        }  # Dictionary to store results for this question

        # Initialize question results with only the question
        question_results = {"question": question}

        for judge_name, judge in judges.items():
            logger.info(f"Evaluating with {judge_name}")

            # Convert the question and answer to a JSON string for LLM input
            json_query_str = json.dumps(original_question_results)
            logger.debug(f"Sending to LLM: {json_query_str}")

            try:
                result = invoke_llm_with_retry(judge, json_query_str)
                logger.info(f"Result: {result}")

                # Remove any prefix like 'json' from the result
                if result.startswith("json"):
                    clean_result = result[len("json") :].strip()
                else:
                    clean_result = result.strip()

                logger.debug(f"Cleaned result for JSON parsing: {clean_result}")

                if clean_result.strip():  # Check if cleaned result is not empty
                    try:
                        result_dict = json.loads(
                            clean_result
                        )  # Parse the cleaned result string into a dictionary
                        question_results[f"{judge_name}_score"] = result_dict.get(
                            "score"
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decode error: {e} for cleaned result: {clean_result}"
                        )
                        question_results[f"{judge_name}_score"] = None
                else:
                    logger.error("Empty result received from LLM after cleaning")
                    question_results[f"{judge_name}_score"] = None
            except Exception as e:
                logger.error(f"Error invoking LLM: {e}")
                question_results[f"{judge_name}_score"] = None

        # Convert scores to numeric, defaulting to 0 if conversion fails
        for key in question_results.keys():
            if key.endswith("_score"):
                try:
                    question_results[key] = float(question_results[key])
                except (ValueError, TypeError):
                    question_results[key] = 0.0

        # Log the question results to verify scores are added
        logger.debug(f"Question results before appending: {question_results}")

        results_list.append(
            question_results
        )  # Append the results for this question to the list

        # Save the results to a CSV file after each question
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(
            os.path.join(REPORTS_DIR, f"evaluation_results_{question_idx+1}.csv"),
            index=False,
        )

    # Log the results list to verify DataFrame construction
    logger.debug(f"Results list: {results_list}")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    questions = results_df["question"]
    metrics = [
        "accuracy_score",
        "completeness_score",
        "relevance_score",
        "traceability_score",
    ]

    # Create a stacked bar chart
    bottom = None
    for metric in metrics:
        plt.bar(questions, results_df[metric], bottom=bottom, label=metric)
        if bottom is None:
            bottom = results_df[metric].copy()
        else:
            bottom += results_df[metric]

    plt.xlabel("Questions")
    plt.ylabel("Scores")
    plt.title("Evaluation Metrics for Each Question")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "evaluation_metrics.png"))
    plt.show()


if __name__ == "__main__":
    main()
