import os
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Never


@contextmanager
def disable_exception_traceback():
    default_value = getattr(
        sys, "tracebacklimit", 1000
    )  # `1000` is a Python's default value
    sys.tracebacklimit = 0
    yield
    sys.tracebacklimit = default_value  # revert changes


class ErrorCode(Enum):
    INSUFFICIENT_CREDITS = "E001"
    INVALID_API_KEY = "E002"
    RATE_LIMIT_EXCEEDED = "E003"
    SERVICE_UNAVAILABLE = "E004"
    MODERATION_CONTENT = "E005"


class ModelError(Exception):
    def __init__(self, error_code: ErrorCode):
        self.error_code = error_code
        self.user_message = "An error occurred while processing your request"

        if error_code == ErrorCode.RATE_LIMIT_EXCEEDED:
            self.user_message = "Service is currently unavailable due to high demand. Please try again later."
        elif error_code == ErrorCode.SERVICE_UNAVAILABLE:
            self.user_message = (
                "Service is temporarily unavailable. Please try again later."
            )
        elif error_code == ErrorCode.MODERATION_CONTENT:
            self.user_message = "The input or output was flagged as sensitive. Please try again with different inputs."

        super().__init__(f"{self.user_message} ({error_code.value})")


ERROR_PATTERNS = {
    ErrorCode.INSUFFICIENT_CREDITS: [
        "credit",
        "fund",
        "billing",
        "payment",
        "subscription",
        "plan",
        "overdue",
    ],
    ErrorCode.MODERATION_CONTENT: [
        "nsfw",
        "explicit",
        "flagged",
        "moderation",
        "risk control",
        "sensitive",
        "violate",
        "moderated",
    ],
    ErrorCode.INVALID_API_KEY: [
        "invalid api key",
        "api key expired",
        "invalid key",
        "unauthorized",
        "authentication failed",
        "auth failed",
    ],
    ErrorCode.RATE_LIMIT_EXCEEDED: [
        "rate limit",
        "too many requests",
        "request limit exceeded",
        "quota exceeded",
        "resource_exhausted",
        "resource exhausted",
        "429",
    ],
    ErrorCode.SERVICE_UNAVAILABLE: [
        "currently unavailable",
        "service unavailable",
        "server error",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "timeout",
        "503",
        "502",
        "504",
    ],
}


def exception_without_traceback(error: Exception) -> Never:
    with disable_exception_traceback():
        raise error from None


def should_show_full_exception():
    model_name = os.getenv("REPLICATE_MODEL_NAME", "")
    deployment_name = os.getenv("REPLICATE_DEPLOYMENT_NAME", None)
    username = os.getenv("REPLICATE_USERNAME", "")

    is_test_model = (
        "test" in model_name.lower() and username == "replicate" and not deployment_name
    )
    is_local_model = model_name == "" and username == ""
    return is_test_model or is_local_model


def check_for_prediction_error(error: Exception) -> Never:
    """
    Check an exception and raise an appropriate ModelError if it matches known error patterns.

    Args:
        error: The exception to check

    Raises:
        ModelError: If the error matches a known pattern
        Exception: The original error if no patterns match
    """

    if should_show_full_exception():
        print(f"Error: {error}")

    error_message = str(error).lower()

    for error_code, patterns in ERROR_PATTERNS.items():
        if any(term in error_message for term in patterns):
            exception_without_traceback(ModelError(error_code))

    exception_without_traceback(error)
