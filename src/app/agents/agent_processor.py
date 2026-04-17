"""
AgentProcessor: manages conversation lifecycle with Microsoft Foundry agents.

Responsibilities:
  - Creating and managing conversation threads via the OpenAI Responses API
  - Dispatching MCP tool calls when the agent requests function execution
  - Extracting text responses from the agent's output

MCP tool wrappers live in mcp_tools.py; FunctionTool definitions live in
tool_definitions.py. This file focuses solely on conversation orchestration.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam
import json
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import time
import logging

from app.agents.mcp_tools import MCP_FUNCTIONS
from app.agents.tool_definitions import get_tools_for_agent

from opentelemetry import trace
from azure.ai.agents.telemetry import trace_function

# Tracing is configured once in chat_app.py (the application entry point).
# Do NOT call configure_azure_monitor() or OpenAIInstrumentor().instrument()
# here — duplicate calls corrupt the OpenTelemetry pipeline.
tracer = trace.get_tracer(__name__)

logger = logging.getLogger(__name__)

# Thread pool for running sync OpenAI SDK calls from async context
_executor = ThreadPoolExecutor(max_workers=8)

# Cache: agent_type -> List[FunctionTool] (populated lazily)
_toolset_cache: Dict[str, list] = {}


class AgentProcessor:
    """Orchestrates a conversation with a Microsoft Foundry agent.

    Each AgentProcessor is bound to a specific agent (by ID and type) and
    manages a conversation thread. When the agent requests a function call,
    the processor dispatches it to the corresponding MCP tool wrapper and
    feeds the result back to the agent.
    """

    def __init__(self, project_client, assistant_id: str, agent_type: str, thread_id=None):
        self.project_client = project_client
        self.agent_id = assistant_id
        self.agent_type = agent_type
        self.thread_id = thread_id

    # -- Streaming (generator) API ----------------------------------------

    def run_conversation_with_text(self, input_message: str = ""):
        """Synchronous generator that yields streamed text chunks."""
        start_time = time.time()
        openai_client = self.project_client.get_openai_client()
        thread_id = self.thread_id

        if thread_id:
            openai_client.conversations.retrieve(conversation_id=thread_id)
            openai_client.conversations.items.create(
                conversation_id=thread_id,
                items=[{"type": "message", "role": "user", "content": input_message}]
            )
        else:
            conversation = openai_client.conversations.create(
                items=[{"role": "user", "content": input_message}]
            )
            thread_id = conversation.id
            self.thread_id = thread_id

        logger.info(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")

        messages = openai_client.responses.create(
            conversation=thread_id,
            extra_body={"agent_reference": {"name": self.agent_id, "type": "agent_reference"}},
            input="",
            stream=True
        )
        for message in messages:
            yield message.response.output_text

        logger.info(f"[TIMELOG] Total run_conversation_with_text time: {time.time() - start_time:.2f}s")

    # -- Async API (preferred) --------------------------------------------

    async def _execute_function_calls(self, message) -> list:
        """Dispatch function calls from agent output to MCP tool handlers."""
        input_list: ResponseInputParam = []
        for item in message.output:
            if item.type != "function_call":
                continue

            with tracer.start_as_current_span(f"tool_call: {item.name}") as span:
                span.set_attribute("tool.name", item.name)
                span.set_attribute("tool.arguments", item.arguments)

                handler = MCP_FUNCTIONS.get(item.name)
                if handler:
                    func_result = await handler(**json.loads(item.arguments))
                else:
                    func_result = {"error": f"Unknown function: {item.name}"}

                output_str = json.dumps({"result": func_result})
                span.set_attribute("tool.output", output_str[:4096])
                logger.info(f"Function {item.name} returned: {func_result}")

            input_list.append(FunctionCallOutput(
                type="function_call_output",
                call_id=item.call_id,
                output=output_str
            ))
        return input_list

    async def _run_conversation(self, input_message: str = "") -> List[str]:
        """Run a single conversation turn, handling any function calls."""
        thread_id = self.thread_id
        start_time = time.time()

        with tracer.start_as_current_span(f"agent_turn: {self.agent_type}") as span:
            span.set_attribute("agent.type", self.agent_type)
            span.set_attribute("agent.id", self.agent_id)
            span.set_attribute("agent.input", input_message[:4096])

            try:
                openai_client = self.project_client.get_openai_client()

                # Create or continue conversation thread
                if thread_id:
                    openai_client.conversations.retrieve(conversation_id=thread_id)
                    openai_client.conversations.items.create(
                        conversation_id=thread_id,
                        items=[{"type": "message", "role": "user", "content": input_message}]
                    )
                else:
                    conversation = openai_client.conversations.create(
                        items=[{"role": "user", "content": input_message}]
                    )
                    thread_id = conversation.id
                    self.thread_id = thread_id

                logger.info(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")

                # Get initial response (sync OpenAI call in thread pool)
                # Copy context so OpenTelemetry spans propagate into the executor thread
                loop = asyncio.get_event_loop()
                ctx = contextvars.copy_context()
                message = await loop.run_in_executor(
                    _executor,
                    lambda: ctx.run(
                        openai_client.responses.create,
                        conversation=thread_id,
                        extra_body={"agent_reference": {"name": self.agent_id, "type": "agent_reference"}},
                        input="",
                        stream=False,
                    )
                )
                logger.info(f"[TIMELOG] Response retrieval took: {time.time() - start_time:.2f}s")

                # If the agent wants to call functions, execute them and get a follow-up
                if len(message.output_text) == 0:
                    logger.info("Agent requested function calls; dispatching to MCP tools")
                    input_list = await self._execute_function_calls(message)

                    ctx2 = contextvars.copy_context()
                    message = await loop.run_in_executor(
                        _executor,
                        lambda: ctx2.run(
                            openai_client.responses.create,
                            input=input_list,
                            previous_response_id=message.id,
                            extra_body={"agent_reference": {"name": self.agent_id, "type": "agent_reference"}},
                        )
                    )

                result_text = self._extract_text(message)
                span.set_attribute("agent.output", result_text[:4096])
                return [result_text]

            except Exception as e:
                logger.error(f"Conversation failed: {e}", exc_info=True)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                return [f"Error processing message: {str(e)}"]

    async def run_conversation_with_text_stream(self, input_message: str = ""):
        """Async generator that yields text responses from the agent."""
        try:
            messages = await self._run_conversation(input_message)
            for msg in messages:
                yield msg
        except Exception as e:
            logger.error(f"Async conversation failed: {e}")
            yield f"Error processing message: {str(e)}"

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _extract_text(message) -> str:
        """Extract text content from an OpenAI Responses API message."""
        content = message.output_text
        if isinstance(content, list):
            text_blocks = []
            for block in content:
                if isinstance(block, dict):
                    text_val = block.get('text', {}).get('value')
                    if text_val:
                        text_blocks.append(text_val)
                elif hasattr(block, 'text') and hasattr(block.text, 'value'):
                    if block.text.value:
                        text_blocks.append(block.text.value)
            if text_blocks:
                return '\n'.join(text_blocks)
        return str(content)

    @classmethod
    def clear_toolset_cache(cls):
        """Clear the toolset cache if needed."""
        global _toolset_cache
        _toolset_cache.clear()

    @classmethod
    def get_cache_stats(cls):
        """Get cache statistics for monitoring."""
        return {
            "toolset_cache_size": len(_toolset_cache),
            "cached_agent_types": list(_toolset_cache.keys())
        }
