import logging
import os
import subprocess
from subprocess import CompletedProcess
from typing import Callable
from langchain_core.tools import tool, BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from .core import OwlCheek, list_roles, load_config
from .db import TOOLS_CONFIG

logger = logging.getLogger("main_logger")


class LocalPythonInterpreter(OwlCheek):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpi_own_message_history = [
            SystemMessage(f"{self.config.system_prompt}")
        ]
        self._script_count = 0

    def _run_system_command(self, user_query: str) -> str:
        """Send a prompt to the model and executes the output as a python script."""
        user_message = HumanMessage(content=user_query)
        self.lpi_own_message_history.append(user_message)

        try:
            model_code_message = self.chat_model.invoke(self.lpi_own_message_history)
            python_script = model_code_message.content
            self.lpi_own_message_history.append(model_code_message)
        except Exception as e:
            error_message = f"Error generating python script: {str(e)}"
            logger.error(error_message)
            return error_message

        try:
            result = self._execute_as_python(python_script)
        except Exception as e:
            error_message = f"Error executing python script: {str(e)}"
            logger.error(error_message)
            return error_message

        return result.stdout

    def _script_label(self):
        return f"SCRIPT {self._script_count:05d}"

    def _next_script(self):
        relative_file_pathname = f"scripts/python/temp_{self._script_count:05d}.py"
        while os.path.exists(relative_file_pathname):
            self._script_count += 1
            return self._next_script()
        return relative_file_pathname

    def _save_to_file(self, script):
        file_relative_pathname = self._next_script()
        with open(file_relative_pathname, "w", encoding="utf-8") as file:
            python_script = f"#{self._script_label()}\n{script}"
            file.write(python_script)
            os.chmod(file_relative_pathname, 0o755)
        return file_relative_pathname

    def _execute_as_python(self, code: str) -> CompletedProcess[str]:
        try:
            file_relative_pathname = self._save_to_file(code)
            #python.exe depends on windows... beurk time to switch to better implementation
            result = subprocess.run(
                ["python.exe", file_relative_pathname],
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            if len(result.stdout) > 0:
                result_message = f"{self._script_label()} - execution completed - Return code : {result.returncode} - STDOUT: {result.stdout}"
                logger.debug(result_message)
                self.lpi_own_message_history.append(HumanMessage(result_message))

            if len(result.stderr) > 0:
                stderr_message = f"{self._script_label()} - Error/Warning message - Return code : {result.returncode} - STDERR: {result.stderr}"
                logger.warning(stderr_message)
                self.lpi_own_message_history.append(HumanMessage(stderr_message))

            return result

        except subprocess.CalledProcessError as cpe:
            error_message = f"{self._script_label()} - execution failed: ProcessError: {str(cpe)} - {cpe.output} - {cpe.stdout} - {cpe.stderr}."
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))
            raise

        except subprocess.TimeoutExpired as toe:
            error_message = f"{self._script_label()} execution failed: TimeoutExpired: {str(toe)} - {toe.output} - {toe.stdout} - {toe.stderr}."
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))
            raise

        except Exception as e:
            error_message = (
                f"{self._script_label()} - execution failed: Exception: {str(e)}"
            )
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))
            raise


class LocalRAGTool(OwlCheek):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
        db_persist_directory = "data/vector"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        self.prompt = PromptTemplate.from_template(self.config.system_prompt)
        self.vector_store = Chroma(
            embedding_function=embeddings, persist_directory=db_persist_directory
        )

    def rag_question(self, question: str) -> str:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.
        """
        retrieved_docs = self.vector_store.similarity_search(query=question, k=4)

        for doc in retrieved_docs:
            logger.debug(f"Document matched: {doc.page_content[:100]} {doc.metadata}")

        docs_content = "\n\n".join(
            doc.page_content.encode("utf-8", errors="ignore").decode("utf-8")
            for doc in retrieved_docs
        )
        message_with_question_and_context = self.prompt.format(
            question=question, context=docs_content
        )
        currated_message_with_question_and_context = (
            message_with_question_and_context.encode("utf-8", errors="ignore").decode(
                "utf-8"
            )
        )
        messages = [SystemMessage(currated_message_with_question_and_context)]
        messages = self.chat_model.invoke(messages)

        logger.debug(f"Raw RAG answer: {messages.content}")
        return messages.content

_lpi_config = load_config("python_interpreter",TOOLS_CONFIG)
_lrt_config = load_config("rag_tool",TOOLS_CONFIG)

_local_python_interpreter = LocalPythonInterpreter(_lpi_config)
_local_rag_tool = LocalRAGTool(_lrt_config)

class ToolBox:



    @tool
    def identify_user_with_password(user_password: str) -> str:
        """
        Checks wether the password is valid.
        A valid password is required and sufficient to identify the user.
        The password is a sequence of words separated by spaces.

        Args:
            user_password: a string containing a sequence of words separated by spaces.
        """
        from .db import get_user_by_password  # Import here to avoid circular imports

        global user_context
        logger.debug(f"calling identify_user_by_password with {user_password}")
        user_data = get_user_by_password(user_password)
        if user_data:
            user_context = f"CONTEXT: {user_data}"
            return (
                "User identified: "
                + user_data["first_name"]
                + " "
                + user_data["last_name"]
            )
        else:
            return "Invalid password"

    @tool
    def activate_mode(mode: str):
        """
        Activates a different owlai mode.
        Args:
            mode: a string containing the mode to activate.
        """

        global focus_role
        #if mode not in list_roles():
        #    return f"Invalid mode: {mode}"
        focus_role = mode
        message = f"Activated {mode} mode"
        logger.debug(message)
        return message

    @tool
    def run_task(task: str):
        """
        Runs a natural language tasks on the local machine.
        Args:
            script: a natural language string describing the command to run.
        """
        toolbox_hook = _local_python_interpreter._run_system_command
        logger.debug(f"Running system task: {task}")
        command_stdout = toolbox_hook(task)
        if command_stdout.endswith("\n"):
            command_stdout = command_stdout[:-1]
        logger.info(f"Command output: {command_stdout}")
        return command_stdout

    @tool
    def play_song(song_name: str = "Fly Away", artist_name: str = "Lenny Kravitz"):
        """
        Plays a song.
        Args:
            song_name: (required) a string containing the name of the song to play.
            artist_name: (required) a string containing the name of the artist of the song to play.
        """
        from .spotify import play_song_on_spotify

        logger.info(f"Playing song: {song_name} by {artist_name}")
        play_song_on_spotify(song_name, artist_name)

    @tool
    def get_answer_from_knowledge_base(question: str):
        """
        Gets an answer from the knowledge base.
        Args:
            question: a string containing the question to answer.
        """
        toolbox_hook_rag_engine = _local_rag_tool.rag_question
        logger.debug(f"Running RAG question: {question} {toolbox_hook_rag_engine}")
        rag_answer = toolbox_hook_rag_engine(question)
        return rag_answer


# Global instances
toolbox = ToolBox()

mapping = {
    "activate": toolbox.activate_mode,
    "identify_user_with_password": toolbox.identify_user_with_password,
    "run_task": toolbox.run_task,
    "play_song": toolbox.play_song,
    "get_answer_from_knowledge_base": toolbox.get_answer_from_knowledge_base,
}

""" Returns the list of callabée tools from the list of tool names based on mapping above (we could have a naming convention to remove the mapping)"""
def get_tools(keys : list[str] ) -> list[Callable] :
    return [mapping[key] for key in keys if key in mapping]

toolbox_hook: Callable = None
toolbox_hook_rag_engine: Callable = None
user_context: str = "CONTEXT: "
focus_role: str = "identification"

def get_focus_role() -> str :
    return focus_role
