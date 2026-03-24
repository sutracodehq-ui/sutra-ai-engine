"""
Stream Filter — reusable async generator wrapper for cleaning LLM output.

Strips Chain-of-Thought blocks (<think>...</think>, [THINKING]...[END THINKING], etc.)
from streamed tokens before they reach the client.
"""

import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# Markers that models use for internal reasoning
COT_START_MARKERS = ["<think>", "[THINKING PROCESS]", "[THINKING]", "[REASONING]"]
COT_END_MARKERS = ["</think>", "[END THINKING PROCESS]", "[END THINKING]", "[END REASONING]"]


async def strip_cot(
    token_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Wrap a token stream and strip any CoT blocks.

    Usage:
        async for token in strip_cot(driver.stream(messages)):
            yield token
    """
    buffer = ""
    in_block = False

    async for token in token_stream:
        buffer += token

        # Check for entering a CoT block
        if not in_block:
            for marker in COT_START_MARKERS:
                if marker in buffer:
                    in_block = True
                    before = buffer.split(marker, 1)[0]
                    if before.strip():
                        yield before
                    buffer = buffer.split(marker, 1)[1]
                    break

        # Check for exiting a CoT block
        if in_block:
            for marker in COT_END_MARKERS:
                if marker in buffer:
                    in_block = False
                    buffer = buffer.split(marker, 1)[1]
                    break
            continue  # Don't yield while inside CoT block

        # Normal token — flush when buffer is large enough or no partial marker match
        if len(buffer) > 50:
            yield buffer
            buffer = ""

    # Flush remaining
    if buffer and not in_block:
        yield buffer
