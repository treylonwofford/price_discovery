"""
Instantiates the agent
"""


import os
from typing import Any, Union, Optional, List
from dotenv import load_dotenv
from src.utils import load_config, logger
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.agents.agent import AgentAction
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


# Instantiate config and environment variables
config = load_config()
load_dotenv()

# Instantiate error handler class
class RetryCallbackHandler(BaseCallbackHandler):
    """
    A callback handler class that attempts to retry a tool execution on error.
    """

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """
        Callback method called when a tool execution encounters an error.

        Args:
            error (Union[Exception, KeyboardInterrupt]): The exception raised during tool execution.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The action to take in response to the error (AgentAction.RERUN or AgentAction.CHANGE_TOOL).
        """
        # Log the error or perform any necessary actions
        logger.error(f"Tool execution error: {str(error)}")

        # Retry logic
        max_retries = 1
        current_retry = kwargs.get("retry_count", 0)
        if current_retry < max_retries:
            logger.info(f"Retrying tool execution (Attempt {current_retry + 1})")
            # Increment retry count
            kwargs["retry_count"] = current_retry + 1
            # Re-run the tool
            return AgentAction.RERUN
        else:
            logger.info("Maximum retries reached. Switching to another tool.")
            return AgentAction.CHANGE_TOOL


class Agent:
    """
    A class representing an AI agent with tools and a language model.

    Args:
        llm_name (str): The name of the language model to use.
        tools (List[Tool]): The list of tools available to the agent.
        prompt (PromptTemplate): The prompt template for the agent.
    """

    def __init__(
        self,
        llm_name: str,
        tools: List[Tool],
        prompt: PromptTemplate,
    ):
        self.llm_name = llm_name
        self.tools = tools
        self.prompt = prompt

    def initialize(self) -> Optional[AgentExecutor]:
        """
        Initialize the agent and return an AgentExecutor instance.

        Returns:
            Optional[AgentExecutor]: The initialized AgentExecutor instance, or None if initialization fails.
        """
        error_handler = RetryCallbackHandler()

        try:
            if self.llm_name.lower() == "mixtral":
                repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id, temperature=config["agents"]["temperature"]
                )

            elif self.llm_name.lower() == "gemini":
                GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3
                )

            agent = create_react_agent(llm, self.tools, self.prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                callbacks=[error_handler],
            )
            return agent_executor
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            return None
