from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor

import requests
import json


def _format_message_content(content: Any) -> Any:
        """Format message content."""
        if content and isinstance(content, list):
            # Remove unexpected block types
            formatted_content = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and "type" in block
                    and block["type"] == "tool_use"
                ):
                    continue
                else:
                    formatted_content.append(block)
        else:
            formatted_content = content

        return formatted_content

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict

def _create_message_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    return [_convert_message_to_dict(m) for m in messages]


class ChatWatsonx(BaseChatModel):
    model_name: str
    """The name of the model"""
    watsonx_api_key: str
    """Bearer token"""

    def _create_chat_result(self, response: requests.Response) -> ChatResult:
        generations = []
        status_code = response.status_code
        resp = response.json()
        if status_code < 200 or status_code >= 300:
            raise ValueError(resp)
        token_usage = {
            "input_token_count": 0,
            "generated_token_count": 0
        }
        for result in resp.get("results"):
            token_usage["input_token_count"] += result.get("input_token_count", 0)
            token_usage["generated_token_count"] += result.get("generated_token_count", 0)

            message = SystemMessage(content=result.get("generated_text", ""))
            generation_info = dict(finish_reason=result.get("stop_reason"))
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": resp.get("model_id", self.model_name),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call the IBM watsonx.ai inference endpoint which then generate the response.
        Args:
            prompts: List of strings (prompts) to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The full ChatResult output.
        Example:
            .. code-block:: python

                response = watsonx_llm.generate(["What is a molecule"])
        """
        print("[DEBUG GENERATE REQUEST]", messages)
        message_dicts = _create_message_dicts(messages)
        response = requests.post(
            "https://bam-api.res.ibm.com/v2/text/chat?version=2024-06-20",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.watsonx_api_key
            },
            json={
                "model_id": self.model_name,
                "messages": message_dicts,
                  "parameters": {
                    "decoding_method": "greedy",
                    "repetition_penalty": 1.05,
                    "stop_sequences": [
                        "<|endoftext|>"
                    ],
                    "include_stop_sequence": False,
                    "min_new_tokens": 1,
                    "max_new_tokens": 4096
                },
            }
        )
        ### Hard code response to save inference for debugging certain responses
        # response = requests.Response()
        # response.status_code = 201
        # dummy = {'id': '3f4ba9ef-fc2f-46f9-b3b3-5ea1c6f9fa7d', 'model_id': 'ibm/granite-20b-code-instruct', 'created_at': '2024-06-22T06:22:42.155Z', 'results': [{'generated_text': '{\n    "action": "DbTablesWithRelevanceScores",\n    "action_input": "what is the total number of albums"\n}', 'generated_token_count': 32, 'input_token_count': 1342, 'stop_reason': 'eos_token'}], 'conversation_id': '5828af53-c35a-438d-b241-d109ecbb6cc4'}
        # response._content = json.dumps(dummy).encode("utf-8")
        print("[DEBUG GENERATE RESPONSE]", response.json())
        return self._create_chat_result(response)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "IBM watsonx.ai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }