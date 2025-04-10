from owlai.config.prompts import PROMPT_CONFIG
from owlai.config.tools import DEFAULT_PARSER_CONFIG, FR_LAW_PARSER_CONFIG
from owlai.config.agents import FRENCH_LAW_QUESTIONS

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

from owlai.services.rag import RAGTool
from owlai.core import OwlAgent
import time
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential


QUESTIONS_TO_TEST = (
    FRENCH_LAW_QUESTIONS["general"]
    + FRENCH_LAW_QUESTIONS["tax"]
    + FRENCH_LAW_QUESTIONS["admin"]
)

PROMPT_JUDGE = """
You are a relevance judge from owlAI. Your task is to evaluate how relevant a given text chunk is to a user query.

Assign a score from -10 to 10 including decimal values based on semantic relevance:

Use the following guidance:

10: Perfect answer; directly and completely answers the query

7 to 9: Highly relevant, contains most or all key points

4 to 6: Moderately relevant; touches on the topic but is incomplete or tangential

1 to 3: Slightly relevant; minor or indirect connection

0: Unrelated but not misleading

-1 to -3: Misleading or off-topic in a subtle way

-4 to -10: Completely irrelevant or factually wrong

Always respond in the specified JSON format. Be concise and only provide the score.
Input format:
{
    "query": "query",
    "chunk": "chunk of text"
}

Output format:
{
   "score": "score between -10 and 10"
}
"""


LLM_JUDGE = {
    "name": "chunk-relevance-judge",
    "version": "1.0",
    "description": "Agent responsible for judging the relevance of a chunk of text with respect to a query",
    "system_prompt": PROMPT_JUDGE,
    "llm_config": {
        "model_provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 4000,
        "temperature": 0.1,
        "context_size": 4000,
        "tools_names": [],
    },
    "default_queries": QUESTIONS_TO_TEST,
}

TOOLS_TO_TEST = {
    "rag-fr-general-law-v1": {
        "name": "rag-fr-general-law-v1",
        "description": "Returns data chunks from french law documents: civil, work, commercial, criminal, residency, social, public health",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "dangvantuan/camembert-base-msmarco",
            "model_kwargs": {"device": "cpu"},
            "embedding_model_kwargs": {},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": False,
            "datastore": {
                "name": "rag-fr-general-law",
                "version": "0.3.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/legal-rag/general",  # Larger dataset
                "parser": FR_LAW_PARSER_CONFIG,
            },
        },
    },
}


TMO_TOOLS = {
    "fr-law-complete": {
        "name": "fr-law-complete",
        "description": "Returns data chunks from french law documents: civil, work, commercial, criminal, residency, social, public health.",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": False,
            "datastore": {
                "name": "rag-fr-law-complete",
                "version": "0.0.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/cache/rag-fr-law-complete",
                "parser": DEFAULT_PARSER_CONFIG,
            },
        },
    },
    "rag-fr-tax-law-v1": {
        "name": "rag-fr-tax-law-v1",
        "description": "Returns data chunks from french tax law documents",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french tax law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": False,
            "datastore": {
                "name": "rag-fr-tax-law",
                "version": "0.3.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/legal-rag/fiscal",  # Larger dataset
                "parser": FR_LAW_PARSER_CONFIG,
            },
        },
    },
}


def safe_correlation(list1, list2):
    """Calculate Pearson correlation safely, handling edge cases like identical values."""
    if len(list1) <= 1 or len(list2) <= 1:
        return 0

    # Calculate means
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)

    # Calculate standard deviations
    var1 = sum((x - mean1) ** 2 for x in list1) / len(list1)
    var2 = sum((x - mean2) ** 2 for x in list2) / len(list2)

    # If either standard deviation is 0, correlation is undefined (return 0)
    if var1 <= 1e-10 or var2 <= 1e-10:
        return 0

    # Calculate correlation
    std1 = var1**0.5
    std2 = var2**0.5
    cov = sum((list1[i] - mean1) * (list2[i] - mean2) for i in range(len(list1))) / len(
        list1
    )

    return cov / (std1 * std2)


def calculate_rank_correlation(rank_list1, rank_list2):
    """Calculate Spearman's rank correlation coefficient."""
    n = len(rank_list1)
    if n <= 1:
        return 0

    # Calculate d² for each pair of ranks
    d_squared_sum = sum((rank_list1[i] - rank_list2[i]) ** 2 for i in range(n))

    # Spearman's formula: rho = 1 - (6 * sum(d²) / (n * (n² - 1)))
    return 1 - (6 * d_squared_sum / (n * (n**2 - 1))) if n > 1 else 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def invoke_llm_with_retry(judge, json_query):
    """Invoke LLM with retry logic for handling timeouts and rate limits."""
    try:
        result = judge.message_invoke(json_query)
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

    from owlai.services.system import sprint
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import hashlib
    import datetime
    import csv

    # Create directories for reports and visualizations
    REPORTS_DIR = "data/evaluation/reports-camembert-rag-fr-general-law-v1"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Create tools to evaluate
    tools = [RAGTool(**itool) for itool in TOOLS_TO_TEST.values()]
    judge = OwlAgent(**LLM_JUDGE)

    for question_idx, question in enumerate(QUESTIONS_TO_TEST):
        print(
            f"Evaluating question {question_idx+1}/{len(QUESTIONS_TO_TEST)}: {question}"
        )

        # List to collect all documents and scores for this question
        all_docs = []

        for tool_to_test in tools:
            print(f"  Using tool: {tool_to_test.name}")
            try:
                full_rag_answer = tool_to_test.get_rag_sources(question)
                reranked_docs_metadata = full_rag_answer["metadata"]["reranked_docs"]
                retrieved_docs_metadata = full_rag_answer["metadata"]["retrieved_docs"]

                reranked_docs_ids = [doc["id"] for doc in reranked_docs_metadata]
                retrieved_docs_ids = [doc["id"] for doc in retrieved_docs_metadata]

                store = tool_to_test._vector_store

                # Judge reranked documents
                from tqdm import tqdm

                for idoc_meta in tqdm(
                    reranked_docs_metadata, desc="Evaluating reranked documents"
                ):
                    store_doc = store.get_by_ids([idoc_meta["id"]])
                    llm_judge_query = {
                        "query": question,
                        "chunk": store_doc[0].page_content,
                    }
                    json_query = json.dumps(llm_judge_query)

                    # Use retry logic for OpenAI API calls
                    try:
                        result = invoke_llm_with_retry(judge, json_query)
                        result_dict = json.loads(result)
                        idoc_meta["judge_score"] = result_dict["score"]

                        # Add to our collection of all docs
                        all_docs.append(
                            {
                                "tool_name": tool_to_test.name,
                                "question": question,
                                "docID": idoc_meta["id"],
                                "judge_score": float(result_dict["score"]),
                                "reranker_score": (
                                    float(idoc_meta["score"])
                                    if idoc_meta["score"] != "Unknown"
                                    else None
                                ),
                                "is_reranked": True,
                            }
                        )
                    except Exception as e:
                        print(
                            f"Failed to evaluate document {idoc_meta['id']}: {str(e)}"
                        )
                        continue

                # Judge retrieved documents
                for idoc_meta in tqdm(
                    retrieved_docs_metadata, desc="Evaluating retrieved documents"
                ):
                    # Skip if already evaluated in reranked set
                    if idoc_meta["id"] in reranked_docs_ids:
                        continue

                    try:
                        store_doc = store.get_by_ids([idoc_meta["id"]])[0]
                        llm_judge_query = {
                            "query": question,
                            "chunk": store_doc.page_content,
                        }
                        json_query = json.dumps(llm_judge_query)

                        # Use retry logic for OpenAI API calls
                        result = invoke_llm_with_retry(judge, json_query)
                        result_dict = json.loads(result)
                        idoc_meta["judge_score"] = result_dict["score"]

                        # Add to our collection of all docs
                        all_docs.append(
                            {
                                "tool_name": tool_to_test.name,
                                "question": question,
                                "docID": idoc_meta["id"],
                                "judge_score": float(result_dict["score"]),
                                "reranker_score": None,  # Not reranked
                                "is_reranked": False,
                            }
                        )
                    except Exception as e:
                        print(
                            f"Failed to evaluate document {idoc_meta['id']}: {str(e)}"
                        )
                        continue

            except Exception as e:
                print(f"Error processing tool {tool_to_test.name}: {str(e)}")
                continue

        # Save results if we have any
        if all_docs:
            # Generate a clean name for the question
            question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save to CSV
            df = pd.DataFrame(all_docs)
            csv_filename = (
                f"{REPORTS_DIR}/question_{question_idx+1}_{question_hash}.csv"
            )
            df.to_csv(csv_filename, index=False)
            print(f"Saved report to {csv_filename}")

            # Generate visualization directly
            try:
                from tests.visualize_results import visualize_question_results

                visualize_question_results(
                    df, question, csv_filename, REPORTS_DIR.replace("reports", "charts")
                )
                print(f"Visualization created for question {question_idx+1}")
            except Exception as e:
                print(f"Failed to create visualization: {str(e)}")
        else:
            print(f"No results to save for question {question_idx+1}")


if __name__ == "__main__":
    main()
