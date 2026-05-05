from . import (
    backend_anthropic,
    backend_codex,
    backend_openai,
    backend_openrouter,
    backend_gemini,
)
from .utils import (
    FunctionSpec,
    OutputType,
    PromptType,
    append_provider_event,
    compile_prompt_to_md,
    write_llm_error_files,
    write_llm_request_files,
    write_llm_response_files,
)
import re
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("aide")
_llm_call_counter = 0
_llm_call_counter_lock = threading.Lock()


def _next_llm_sequence_id() -> int:
    global _llm_call_counter
    with _llm_call_counter_lock:
        _llm_call_counter += 1
        return _llm_call_counter


def determine_provider(model: str) -> str:
    # Check if model matches OpenAI patterns first
    if re.match(r"^(gpt-.*|o\d+(-.*)?|codex-mini-latest)$", model):
        return "codex"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gemini"
    # If OPENAI_BASE_URL is set, use openai provider for non-standard models
    elif os.getenv("OPENAI_BASE_URL"):
        return "openai"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "codex": backend_codex.query,
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "openrouter": backend_openrouter.query,
    "gemini": backend_gemini.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    llm_log_dir = model_kwargs.pop("llm_log_dir", None)
    llm_log_prefix = model_kwargs.pop("llm_log_prefix", "")
    llm_log_context = model_kwargs.pop("llm_log_context", None) or {}

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    compiled_system_message = (
        compile_prompt_to_md(system_message) if system_message else None
    )
    compiled_user_message = compile_prompt_to_md(user_message) if user_message else None

    sequence_id = _next_llm_sequence_id()
    log_dir = Path(llm_log_dir) if llm_log_dir is not None else None
    log_context = {
        "phase": llm_log_context.get("phase", llm_log_prefix or "llm"),
        "provider": provider,
        "model": model,
        "sequence_id": sequence_id,
        **llm_log_context,
    }

    request_payload = {
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_kwargs": model_kwargs,
        "func_spec": func_spec.to_dict() if func_spec is not None else None,
        "system_message": compiled_system_message,
        "user_message": compiled_user_message,
    }
    write_llm_request_files(
        log_dir=log_dir,
        prefix=llm_log_prefix,
        context=log_context,
        request_payload=request_payload,
        system_message=compiled_system_message,
        user_message=compiled_user_message,
    )

    try:
        provider_kwargs = dict(model_kwargs)
        if provider == "codex" and log_dir is not None:
            provider_kwargs["llm_log_dir"] = str(log_dir)
            provider_kwargs["llm_log_prefix"] = llm_log_prefix
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=compiled_system_message,
            user_message=compiled_user_message,
            func_spec=func_spec,
            **provider_kwargs,
        )
    except BaseException as exc:
        error_payload = {
            "status": "failed",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
        write_llm_error_files(
            log_dir=log_dir,
            prefix=llm_log_prefix,
            context=log_context,
            error_payload=error_payload,
        )
        if provider != "codex":
            append_provider_event(
                log_dir=log_dir,
                prefix=llm_log_prefix,
                event={
                    "event": "error",
                    "provider": provider,
                    "sequence_id": sequence_id,
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
        raise

    response_payload = {
        "status": "completed",
        "output": output,
        "request_time_seconds": req_time,
        "input_tokens": in_tok_count,
        "output_tokens": out_tok_count,
        "info": info,
    }
    write_llm_response_files(
        log_dir=log_dir,
        prefix=llm_log_prefix,
        context=log_context,
        response_payload=response_payload,
        output=output,
    )
    if provider != "codex":
        append_provider_event(
            log_dir=log_dir,
            prefix=llm_log_prefix,
            event={
                "event": "completed",
                "provider": provider,
                "sequence_id": sequence_id,
                "request_time_seconds": req_time,
                "input_tokens": in_tok_count,
                "output_tokens": out_tok_count,
            },
        )

    return output
