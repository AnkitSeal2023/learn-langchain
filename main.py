
from typing import List, Generic

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama

from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")

   
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

llm = ChatOllama(model="qwen3.5:9b", reasoning=False, num_ctx=32768)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse))


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    structured_data: AgentResponse = result["structured_response"]
    print(f"structured data:\n =============\n {structured_data}\n ================\n")
    print(f"result={result}")


if __name__ == "__main__":
    main()
