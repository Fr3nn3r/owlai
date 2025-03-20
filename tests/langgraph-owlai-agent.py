class OwlAIAgent:

    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        context_size: int = 4096,
        tools: List[BaseTool] = [],
        system_prompt: Optional[str] = None,
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_size = context_size
        self.tools = tools
        self.system_prompt = system_prompt
        self.chat_model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.state_memory = MemorySaver()
        self.state_config = cast(
            RunnableConfig, {"configurable": {"thread_id": str(id(self))}}
        )

        self.agent_graph = create_react_agent(
            self.chat_model,
            self.tools,
            prompt=(
                SystemMessage(content="")
                if self.system_prompt is None
                else SystemMessage(content=self.system_prompt)
            ),
            checkpointer=self.state_memory,
        )

    def invoke(self, message: str) -> str:
        graph_response = self.agent_graph.invoke(
            {"messages": [HumanMessage(message)]}, self.state_config
        )

        if logger.isEnabledFor(logging.DEBUG):
            sprint(graph_response)

        return self._return_last_message_content(graph_response)

    def _return_last_message_content(self, response: Dict[str, Any]) -> str:
        return response["messages"][-1].content

    def print__message_history(self):
        state = self.agent_graph.get_state(self.state_config)
        for index, message in enumerate(state.values["messages"]):
            logger.info(
                f"Message #{index} type: '{message.type}' content: '{ (message.content[:100] + '...' if len(message.content) > 100 else message.content)}'"
            )
            if logger.isEnabledFor(logging.DEBUG):
                sprint(message)

    def print_message_metadata(self):
        state = self.agent_graph.get_state(self.state_config)
        for index, message in enumerate(state.values["messages"]):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        logger.info(f"System prompt: '{self.system_prompt}'")

    def reset__message_history(self):
        logger.warning("Resetting message history not supported")

    def run_tests(self):
        logger.warning("Running tests not supported (NOT TO BE MANAGED HERE)")

    def print_info(self):
        logger.info(
            f"model-provider='{self.model_provider}', model-name='{self.model_name}', tools='{', '.join([t.name for t in self.tools])}'"
        )

    # NOT SURE WHAT THIS IS...
    # Add proper resource cleanup (???):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.warning("Cleanup not supported (???)")
