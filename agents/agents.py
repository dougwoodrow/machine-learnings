from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama/deepseek-r1", # change the r1 model name accordingly
    api_base="http://127.0.0.1:11434",
    api_key='lm-studio',
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, add_base_tools=True)

agent.run(
    "What is 2+2?",
)