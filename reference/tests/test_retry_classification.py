"""Tests for non-retryable model-error classification.

A 4xx response is fully determined by the request: retrying the identical
request can only reproduce the failure. Before these names were listed, a
single typo'd model name produced 3x retries per node (measured: 11
tracebacks, 57s wall clock, for one --model typo).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.config import _NON_RETRYABLE_EXC_NAMES


class NonRetryableNamesTest(unittest.TestCase):
    """The litellm 4xx families must be classified as non-retryable."""

    LITELLM_4XX = (
        "NotFoundError",
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
        "ContentPolicyViolationError",
        "UnsupportedParamsError",
    )

    def test_4xx_families_are_non_retryable(self):
        for name in self.LITELLM_4XX:
            self.assertIn(name, _NON_RETRYABLE_EXC_NAMES, name)

    def test_budget_and_overflow_names_still_present(self):
        # The original entries must survive the extension: retrying a
        # budget stop or a context overflow is equally pointless.
        for name in (
            "BudgetExceededError",
            "LlmCallsLimitExceededError",
            "MantisAuthError",
            "ContextBudgetExceededError",
            "ContextWindowExceededError",
        ):
            self.assertIn(name, _NON_RETRYABLE_EXC_NAMES, name)


class AdkNodeRetryPatchTest(unittest.TestCase):
    """The ADK node-retry monkeypatch must honor the extended list, both for
    the exception itself and for a wrapped __cause__."""

    def setUp(self):
        try:
            import google.adk.workflow.utils._retry_utils as adk_retry
        except Exception:
            self.skipTest("google.adk retry utils not importable here")
        self.should_retry = adk_retry._should_retry_node

    def _exc(self, name: str) -> BaseException:
        return type(name, (Exception,), {})("synthetic")

    def test_not_found_is_not_retried(self):
        self.assertFalse(
            self.should_retry(self._exc("NotFoundError"), MagicMock(), MagicMock())
        )

    def test_wrapped_cause_is_not_retried(self):
        outer = RuntimeError("wrapper")
        outer.__cause__ = self._exc("BadRequestError")
        self.assertFalse(self.should_retry(outer, MagicMock(), MagicMock()))


if __name__ == "__main__":
    unittest.main()
