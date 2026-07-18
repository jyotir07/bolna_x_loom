"""Bolna-facing OpenAI-compatible chat endpoint.

Bolna points its "custom LLM" at `POST /v1/chat/completions` and talks the
OpenAI protocol. We route the turn through Loom and answer in OpenAI shape.
Loom has no streaming API, so when `stream=true` we emit a valid OpenAI SSE
stream assembled from the complete response — correct on the wire, just not
token-incremental.
"""

from typing import AsyncIterator

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agents.conversation import handle_turn
from backend.models.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    CompletionUsage,
    DeltaMessage,
    ResponseMessage,
)
from backend.services.loom import LLMServiceError
from backend.services.logger import get_logger, log_turn

router = APIRouter()
logger = get_logger("bolna_x_loom.chat")


def _error_response(exc: LLMServiceError) -> JSONResponse:
    # 502 for upstream/retryable failures, 500 for our-side/config failures.
    status = 502 if exc.retryable else 500
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": str(exc),
                "type": "upstream_error" if exc.retryable else "server_error",
                "provider": exc.provider,
            }
        },
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_conversation_id: str | None = Header(default=None),
):
    if not any(m.role == "user" for m in request.messages):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no user message provided", "type": "invalid_request"}},
        )

    conversation_id = x_conversation_id or request.user

    request_params: dict = {}
    if request.temperature is not None:
        request_params["temperature"] = request.temperature
    if request.max_tokens is not None:
        request_params["max_tokens"] = request.max_tokens

    try:
        result, decision = await handle_turn(
            request.messages,
            conversation_id=conversation_id,
            request_params=request_params,
        )
    except LLMServiceError as exc:
        logger.warning("turn failed cid=%s: %s", conversation_id or "-", exc)
        return _error_response(exc)

    log_turn(logger, decision=decision, result=result, conversation_id=conversation_id)

    if request.stream:
        return StreamingResponse(
            _stream_chunks(result.text, request.model),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        model=request.model,
        choices=[Choice(message=ResponseMessage(content=result.text))],
        usage=CompletionUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.input_tokens + result.usage.output_tokens,
        ),
    )


async def _stream_chunks(text: str, model: str) -> AsyncIterator[str]:
    def sse(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    # 1) role delta, 2) the content, 3) stop.
    yield sse(ChatCompletionChunk(model=model, choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))]))
    yield sse(ChatCompletionChunk(model=model, choices=[ChunkChoice(delta=DeltaMessage(content=text))]))
    yield sse(ChatCompletionChunk(model=model, choices=[ChunkChoice(delta=DeltaMessage(), finish_reason="stop")]))
    yield "data: [DONE]\n\n"
