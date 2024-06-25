import json
import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from dataherald.utils.agent_prompts import FORMAT_INSTRUCTIONS_GRANITE

class GraniteAgentOutputParser(AgentOutputParser):
    """Output parser for granite chat agent."""

    # pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    pattern = re.compile(r"(```(?:json\s+)?)?({(\W.*?)})(```)?", re.DOTALL)
    """Regex pattern to parse the output."""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return FORMAT_INSTRUCTIONS_GRANITE

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            cleaned_output = text.strip()
            action_match = self.pattern.search(cleaned_output)
            if action_match is not None:
                response = json.loads(action_match.group(2).strip(), strict=False)
                if isinstance(response, list):
                    # Got multiple action responses
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                else:
                    return AgentAction(
                        response["action"], response.get("action_input", {}), text
                    )
            elif cleaned_output == "I don't know":
                return AgentFinish({"output": text}, text)
            else:
                return AgentAction("ThinkAction", cleaned_output, text)
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "granite_parser"
