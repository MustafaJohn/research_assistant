"""
tools/call_llm.py

Gemini wrapper with:
  - Model fallback chain: gemini-2.5-pro → gemini-1.5-flash
  - Clean, user-facing error messages (no raw API objects)
  - Exponential backoff on 503/429 before falling back

Why fallback?
  gemini-2.5-pro is the newest model and regularly 503s under high demand.
  gemini-1.5-flash is stable, fast, and has much higher rate limits.
  The output quality difference is negligible for summarization tasks.
"""

import os
import time
import logging
from typing import Optional

import google.genai as genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

# Ordered by preference — first available and non-overloaded wins
_MODEL_CHAIN = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# How long to wait before retrying the same model (seconds)
_RETRY_DELAYS = [2, 4]

# Status codes that mean "overloaded, try again or fall back"
_RETRYABLE_CODES = {503, 429, 500}

# Human-readable messages for known error codes
_ERROR_MESSAGES = {
    503: "The AI model is experiencing high demand right now. Please try again in a moment.",
    429: "Rate limit reached. Please wait a few seconds and try again.",
    500: "The AI service returned an internal error. Retrying with a fallback model.",
    401: "Invalid Gemini API key. Please check your GEMINI_API_KEY environment variable.",
    400: "The request was malformed. This is likely a prompt formatting issue.",
}


def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call Gemini with automatic model fallback and clean error handling.

    Args:
        prompt: Full prompt string
        model:  Optional model override — skips the fallback chain if set

    Returns:
        Generated text string

    Raises:
        ValueError: API key not configured
        RuntimeError: All models failed — includes a clean user-facing message
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    client = genai.Client(api_key=api_key)

    # If a specific model is requested, use it directly with no fallback
    if model:
        return _call_single(client, model, prompt)

    # Otherwise walk the fallback chain
    last_error = None
    for i, m in enumerate(_MODEL_CHAIN):
        try:
            logger.info("[llm] Calling %s", m)
            result = _call_single(client, m, prompt)
            if i > 0:
                logger.info("[llm] Fallback to %s succeeded", m)
            return result

        except RuntimeError as exc:
            last_error = exc
            code = getattr(exc, "status_code", None)

            if code in _RETRYABLE_CODES and i < len(_MODEL_CHAIN) - 1:
                # Brief wait before trying next model
                wait = _RETRY_DELAYS[min(i, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    "[llm] %s returned %s — waiting %ds then trying %s",
                    m, code, wait, _MODEL_CHAIN[i + 1]
                )
                time.sleep(wait)
                continue

            # Non-retryable error or last model — raise immediately
            raise

    raise last_error or RuntimeError("All Gemini models failed.")


def _call_single(client: genai.Client, model: str, prompt: str) -> str:
    """
    Make a single Gemini call. Raises RuntimeError with a clean message on failure.
    Attaches status_code so the caller can decide whether to retry.
    """
    try:
        response = client.models.generate_content(
            model    = model,
            contents = prompt,
        )
        return response.text

    except genai_errors.ClientError as exc:
        code    = _extract_code(exc)
        message = _ERROR_MESSAGES.get(code, f"Request error (code {code}).")
        logger.error("[llm] %s ClientError %s: %s", model, code, exc)
        err = RuntimeError(message)
        err.status_code = code
        raise err

    except genai_errors.ServerError as exc:
        code    = _extract_code(exc) or 503
        message = _ERROR_MESSAGES.get(code, "The AI service is temporarily unavailable. Please try again.")
        logger.error("[llm] %s ServerError %s: %s", model, code, exc)
        err = RuntimeError(message)
        err.status_code = code
        raise err

    except Exception as exc:
        # Catch-all for unexpected errors (network issues, etc.)
        logger.error("[llm] %s unexpected error: %s", model, exc)
        err = RuntimeError(
            "An unexpected error occurred while contacting the AI service. "
            "Please check your connection and try again."
        )
        err.status_code = None
        raise err


def _extract_code(exc: Exception) -> Optional[int]:
    """Pull the HTTP status code out of a Gemini exception."""
    # google-genai attaches it in different places depending on version
    for attr in ("code", "status_code", "grpc_status_code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Try parsing from the string representation
    msg = str(exc)
    for code in _RETRYABLE_CODES | {401, 400, 404}:
        if str(code) in msg:
            return code
    return None
