import os
import subprocess
from subprocess import CompletedProcess
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from .core import OwlAgent

logger = logging.getLogger("interpreter")


class OwlSystemInput(BaseModel):

    query: str = Field(
        description="a natural language string describing the command to run on the local machine"
    )


class LocalPythonInterpreter(OwlAgent):

    _script_count = 0

    def _run_system_command(self, user_query: str) -> str:
        """Send a prompt to the model and executes the output as a python script."""
        self._message_history = [SystemMessage(f"{self.system_prompt}")]
        user_message = HumanMessage(content=user_query)
        self._message_history.append(user_message)

        try:
            model_code_message = self.chat_model.invoke(self._message_history)
            python_script = model_code_message.content
            self._message_history.append(model_code_message)
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
            # python.exe depends on windows... beurk time to switch to better implementation
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
                self._message_history.append(HumanMessage(result_message))

            if len(result.stderr) > 0:
                stderr_message = f"{self._script_label()} - Error/Warning message - Return code : {result.returncode} - STDERR: {result.stderr}"
                logger.warning(stderr_message)
                self._message_history.append(HumanMessage(stderr_message))

            return result

        except subprocess.CalledProcessError as cpe:
            error_message = f"{self._script_label()} - execution failed: ProcessError: {str(cpe)} - {cpe.output} - {cpe.stdout} - {cpe.stderr}."
            logger.error(error_message)
            self._message_history.append(HumanMessage(error_message))
            raise

        except subprocess.TimeoutExpired as toe:
            error_message = f"{self._script_label()} execution failed: TimeoutExpired: {str(toe)} - {toe.output} - {toe.stdout} - {toe.stderr}."
            logger.error(error_message)
            self._message_history.append(HumanMessage(error_message))
            raise

        except Exception as e:
            error_message = (
                f"{self._script_label()} - execution failed: Exception: {str(e)}"
            )
            logger.error(error_message)
            self._message_history.append(HumanMessage(error_message))
            raise


class OwlSystemInterpreter(BaseTool, LocalPythonInterpreter):
    """Tool that execute system commandsand gets back json"""

    name: str = "owl_system_interpreter"
    description: str = "Runs a natural language tasks on the local machine"
    args_schema: Type[BaseModel] = OwlSystemInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool."""
        return self._run_system_command(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool asynchronously."""
        return self._run_system_command(query)
