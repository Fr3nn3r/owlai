if __name__ == "__main__":
    # this prevents reloading when multi processing is used

    import os
    import time
    import transformers

    import logging
    import logging.config
    import yaml
    import json

    from owlai.rag import LocalRAGTool
    from owlai.db import TOOLS_CONFIG

    from tqdm import tqdm

    from owlai.owlsys import get_system_info

    def load_logger_config():
        with open("logging.yaml", "r") as logger_config:
            config = yaml.safe_load(logger_config)
            logging.config.dictConfig(config)

    transformers.logging.set_verbosity_error()

    load_logger_config()

    logger = logging.getLogger("ragtool")

    def main():

        datasets = [
            {
                "input_data_folder": "data/dataset-0000",
                "questions": [
                    "Who is Rich Draves?",
                    "What happened to Paul Graham in the fall of 1992?",
                    "What was the last result of the AC Milan soccer team?",
                    "What did Paul Graham do growing up?",
                    "What did Paul Graham do during his school days?",
                    "What languages did Paul Graham use?",
                    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
                    "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
                    "What is the color of henry the fourth white horse?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0001",
                "questions": [
                    "Who is Naruto?",
                    "Provide details about Orochimaru",
                    "Who is the strongest ninja in the world?",
                    "How is sasuke's personality",
                    "Who is the sensei of naruto?",
                    "What is a sharingan?",
                    "What is the akatsuki?",
                    "Who is the first Hokage?",
                    "What was the last result of the AC Milan soccer team?",
                    "What is the color of henry the fourth white horse?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0003",
                "questions": [
                    "What is Arakis?",
                    "Who is Duncan Idaho?",
                    "Who is the traitor?",
                    "What are the powers of the Bene Gesserit?",
                    "How does Paul defeat the Emperor?",
                ],
            },
        ]

        rag_tool = LocalRAGTool(**TOOLS_CONFIG["owl_memory_tool"])

        # Start of the main function ######################################################
        EMBEDDING_MODEL_NAME = rag_tool._embeddings.model_name

        embedding_model = rag_tool._embeddings

        dataset = datasets[2]

        input_data_folder = dataset["input_data_folder"]

        KNOWLEDGE_VECTOR_DATABASE = rag_tool.load_or_create_vector_store(
            input_data_folder, embedding_model
        )

        questions = dataset["questions"]

        qa_results = []
        qa_results.append(get_system_info())

        qa_results.append(TOOLS_CONFIG["owl_memory_tool"])

        from datetime import datetime

        # Get current date and time
        now = datetime.now()

        # Format as YYYY-MM-DD-HH-MM-SSS
        formatted_date = now.strftime("%Y-%m-%d-%Hh%Mm%f")[
            :-3
        ]  # Trim last 3 digits for milliseconds

        qa_results.append({"test_date": formatted_date})

        test_report_filename = f"{formatted_date}-qa_results.json"

        for question in questions:
            answer = rag_tool.rag_question(question)
            logger.info(f"USER QUERY : {question}")
            logger.info(f"ANSWER : {answer}")

            qa_results.append({"question": question, "answer": answer})

        # Save results to JSON file
        output_file = os.path.join(input_data_folder, test_report_filename)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
        # logger.info(f"SOURCE DOCS : {relevant_docs}")

    print("Application started")
    from multiprocessing import freeze_support
    from dotenv import load_dotenv

    load_dotenv()
    freeze_support()
    main()
