import json
import re
from typing import Any, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableSerializable
from langchain.agents.agent import AgentOutputParser
from dataherald.utils.agent_prompts import FORMAT_INSTRUCTIONS

class GraniteAgentOutputParser(AgentOutputParser):
    """Output parser for granite chat agent."""

    # pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    pattern = re.compile(r"(```(?:json\s+)?)?({(\W.*?)})(```)?", re.DOTALL)
    """Regex pattern to parse the output."""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            cleaned_output = text.strip()
            action_match = self.pattern.search(cleaned_output)
            if action_match is not None:
                response = json.loads(action_match.group(2).strip(), strict=False)
                if isinstance(response, list):
                    # Got multiple action responses
                    response = response[0]
                print("[DEBUG PARSE]", response)
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

class GraniteAgentOutputParserWithRetries(AgentOutputParser):
    """Output parser with retries for the structured chat agent."""

    base_parser: AgentOutputParser = Field(default_factory=GraniteAgentOutputParser)
    """The base parser to use."""
    retry_chain: Union[RunnableSerializable, Any]
    """The RunnableSerializable to use to retry the completion."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # try:
        #     if self.output_fixing_parser is not None:
        #         parsed_obj: Union[
        #             AgentAction, AgentFinish
        #         ] = self.output_fixing_parser.parse(text)
        #     else:
        #         parsed_obj = self.base_parser.parse(text)
        #     print("[DEBUG PARSE RETRY]", parsed_obj)
        #     return parsed_obj
        # except Exception as e:
        #     raise OutputParserException(f"Could not parse LLM output: {text}") from e

        retries = 0
        while retries <= self.max_retries:
            try:
                return self.base_parser.parse(text)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    try:
                        text = self.retry_chain.invoke(
                            dict(
                                instructions=self.base_parser.get_format_instructions(),
                                input=text,
                                error=repr(e),
                            )
                        )
                    except (NotImplementedError, AttributeError):
                        # Case: self.parser does not have get_format_instructions
                        text = self.retry_chain.invoke(
                            dict(
                                agent_scratchpad=text
                            )
                        )
                    text = self.retry_chain.invoke()
        raise OutputParserException(f"Could not parse LLM output: {text}")

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        base_parser: Optional[AgentOutputParser] = None,
    ) -> "GraniteAgentOutputParserWithRetries":
        base_parser = base_parser or GraniteAgentOutputParser()
        chain = llm
        return cls(base_parser=base_parser, retry_chain=chain)

    @property
    def _type(self) -> str:
        return "granite_parser_with_retries"