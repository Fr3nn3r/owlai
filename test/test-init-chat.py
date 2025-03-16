from langchain.chat_models import init_chat_model


chat_model = init_chat_model(
    model="llama3.2:latest",
    model_provider="ollama",
    temperature=0.1,
    max_tokens=2048,
)

print(chat_model.invoke("Hello, how are you?"))

if False:
    from langchain_ollama import ChatOllama

    chat_model = ChatOllama(
        model="llama3.2:latest",  # Ensure the correct model name
        temperature=0.1,
        max_tokens=2048,
    )

    response = chat_model.invoke("Hello, how are you?")
    print(response)
