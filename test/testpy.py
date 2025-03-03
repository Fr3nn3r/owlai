# let's test the local python interpreter

from owlai import LocalPythonInterpreter, get_system_prompt_by_role

lpi = LocalPythonInterpreter(
        role="command_manager",
        implementation="anthropic",
        model_name="claude-3-7-sonnet-20250219",
        temperature=0.9,
        max_tokens=2048,
        max_context_tokens=4096,
        tools=[],
        system_prompt=get_system_prompt_by_role("command_manager"),
    )

lpi.run_tests()

