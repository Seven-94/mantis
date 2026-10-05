"""Unit tests for ResilientLiteLLMClient streaming support (stream=True).

The resilience boundary under test:
- Every failure up to and including the FIRST chunk (connect errors, stream-open
  errors, first-chunk errors) re-enters the full-jitter backoff / auth-refresh
  loop, with the patience clock shared across restarts.
- A failure AFTER the first chunk raises MantisStreamInterruptedError
  (chunks_yielded, original_exception) instead of a transparent retry, because
  re-sampling a nondeterministic model would splice two different completions.
- The context budget guard runs eagerly at call time, not at first iteration.
- The sync completion() path mirrors the async path.
"""

import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

import litellm

import core.config as config_module
from core.config import (
    ContextBudgetExceededError,
    MantisAuthError,
    MantisStreamInterruptedError,
    ResilientLiteLLMClient,
)

# Keep retries fast and deterministic-ish: ~1ms delays, generous patience so
# only the explicit patience test exercises exhaustion.
_FAST_RETRY_ENV = {
    "MANTIS_LLM_RETRY_INITIAL_DELAY": "0.001",
    "MANTIS_LLM_RETRY_MAX_DELAY": "0.002",
    "MANTIS_LLM_MIN_OFFSET": "0.0",
    "MANTIS_LLM_RETRY_BACKOFF": "1.0",
    "MANTIS_LLM_MAX_PATIENCE_SECONDS": "30.0",
}


class _Transient429(Exception):
    """Shaped like a retryable rate-limit error (status_code drives classification)."""

    status_code = 429


class _Auth401(Exception):
    """Shaped like an authentication failure."""

    status_code = 401


class _FakeAsyncStream:
    """Async-iterable stream that can fail before yielding its index-th chunk."""

    def __init__(self, chunks, fail_before_index=None, exc=None):
        self._chunks = list(chunks)
        self._index = 0
        self._fail_before_index = fail_before_index
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fail_before_index is not None and self._index == self._fail_before_index:
            raise self._exc
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _FakeSyncStream:
    """Sync mirror of _FakeAsyncStream."""

    def __init__(self, chunks, fail_before_index=None, exc=None):
        self._chunks = list(chunks)
        self._index = 0
        self._fail_before_index = fail_before_index
        self._exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        if self._fail_before_index is not None and self._index == self._fail_before_index:
            raise self._exc
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


async def _collect(stream):
    return [chunk async for chunk in stream]


_MESSAGES = [{"role": "user", "content": "hi"}]


class TestResilientStreaming(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = ResilientLiteLLMClient()
        env_patch = patch.dict(os.environ, _FAST_RETRY_ENV)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    # ------------------------------------------------------------------
    # Signature parity
    # ------------------------------------------------------------------

    def test_both_paths_expose_stream_parameter(self):
        for method in (ResilientLiteLLMClient.acompletion, ResilientLiteLLMClient.completion):
            params = inspect.signature(method).parameters
            self.assertIn("stream", params)
            self.assertIs(params["stream"].default, False)

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def test_async_nonstream_path_unchanged(self):
        sentinel = object()
        seen = {}

        async def fake_acompletion(**kw):
            seen.update(kw)
            return sentinel

        with patch.object(litellm, "acompletion", fake_acompletion):
            result = await self.client.acompletion(model="m", messages=_MESSAGES)

        self.assertIs(result, sentinel)
        # The non-stream call must stay byte-identical to the pre-stream-support
        # request: no stream kwarg invented on behalf of the caller.
        self.assertNotIn("stream", seen)

    async def test_async_stream_happy_path(self):
        calls = []

        async def fake_acompletion(**kw):
            calls.append(kw)
            return _FakeAsyncStream(["c0", "c1", "c2"])

        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            got = await _collect(stream)

        self.assertEqual(got, ["c0", "c1", "c2"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["stream"])

    async def test_async_stream_empty_stream_is_success(self):
        async def fake_acompletion(**kw):
            return _FakeAsyncStream([])

        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            self.assertEqual(await _collect(stream), [])

    async def test_async_retryable_error_at_connect_is_retried(self):
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Transient429("rate limited at connect")
            return _FakeAsyncStream(["a", "b"])

        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            got = await _collect(stream)

        self.assertEqual(got, ["a", "b"])
        self.assertEqual(calls["n"], 2)

    async def test_async_error_before_first_chunk_is_retried(self):
        # The retry window must extend past "stream opened" to "first chunk
        # delivered": a 429 on the first __anext__ is still a zero-output failure.
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeAsyncStream(["a", "b"], fail_before_index=0, exc=_Transient429("quota"))
            return _FakeAsyncStream(["a", "b"])

        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            got = await _collect(stream)

        self.assertEqual(got, ["a", "b"])
        self.assertEqual(calls["n"], 2)

    async def test_async_midstream_failure_raises_without_retry(self):
        # Even a retryable (429-shaped) error must NOT restart the stream once a
        # chunk has been delivered.
        inner = _Transient429("reset mid-stream")
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            return _FakeAsyncStream(["a", "b", "c"], fail_before_index=2, exc=inner)

        got = []
        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            with self.assertRaises(MantisStreamInterruptedError) as cm:
                async for chunk in stream:
                    got.append(chunk)

        self.assertEqual(got, ["a", "b"])
        self.assertEqual(cm.exception.chunks_yielded, 2)
        self.assertIs(cm.exception.original_exception, inner)
        self.assertIs(cm.exception.__cause__, inner)
        self.assertEqual(calls["n"], 1)

    async def test_async_nonretryable_error_at_connect_raises_immediately(self):
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            raise ValueError("deterministic bad request")

        with patch.object(litellm, "acompletion", fake_acompletion):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            with self.assertRaises(ValueError):
                await _collect(stream)

        self.assertEqual(calls["n"], 1)

    async def test_async_patience_exhaustion_reraises_original_error(self):
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            raise _Transient429("still rate limited")

        with patch.dict(os.environ, {"MANTIS_LLM_MAX_PATIENCE_SECONDS": "0.0"}):
            with patch.object(litellm, "acompletion", fake_acompletion):
                stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
                with self.assertRaises(_Transient429):
                    await _collect(stream)

        self.assertEqual(calls["n"], 1)

    async def test_async_auth_refresh_covers_stream_open(self):
        calls = {"n": 0}

        async def fake_acompletion(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Auth401("token expired")
            return _FakeAsyncStream(["x"])

        with patch.object(litellm, "acompletion", fake_acompletion), \
                patch.object(config_module, "is_token_refreshable_auth_error", return_value=True), \
                patch.object(config_module, "try_refresh_auth", return_value=True):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            got = await _collect(stream)

        self.assertEqual(got, ["x"])
        self.assertEqual(calls["n"], 2)

    async def test_async_unrefreshable_auth_error_wraps_as_mantis_auth_error(self):
        async def fake_acompletion(**kw):
            raise _Auth401("token expired, no refresh available")

        with patch.object(litellm, "acompletion", fake_acompletion), \
                patch.object(config_module, "try_refresh_auth", return_value=False):
            stream = await self.client.acompletion(model="m", messages=_MESSAGES, stream=True)
            with self.assertRaises(MantisAuthError):
                await _collect(stream)

    async def test_async_context_budget_enforced_eagerly_at_call_time(self):
        # The guard must fire at `await acompletion(...)` -- NOT lazily at first
        # iteration -- so an oversized payload is refused before any dispatch.
        async def fake_acompletion(**kw):  # pragma: no cover - must never run
            raise AssertionError("dispatch should have been refused by the budget guard")

        oversized = [{"role": "user", "content": "x" * 4000}]
        with patch.object(litellm, "acompletion", fake_acompletion), \
                patch.object(litellm, "get_max_input_tokens", return_value=10, create=True):
            with self.assertRaises(ContextBudgetExceededError):
                await self.client.acompletion(model="m", messages=oversized, stream=True)

    # ------------------------------------------------------------------
    # Sync path (mirror)
    # ------------------------------------------------------------------

    def test_sync_stream_happy_path(self):
        calls = []

        def fake_completion(**kw):
            calls.append(kw)
            return _FakeSyncStream(["s0", "s1"])

        with patch.object(litellm, "completion", fake_completion):
            got = list(self.client.completion(model="m", messages=_MESSAGES, stream=True))

        self.assertEqual(got, ["s0", "s1"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["stream"])

    def test_sync_error_before_first_chunk_is_retried(self):
        calls = {"n": 0}

        def fake_completion(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeSyncStream(["a"], fail_before_index=0, exc=_Transient429("quota"))
            return _FakeSyncStream(["a"])

        with patch.object(litellm, "completion", fake_completion):
            got = list(self.client.completion(model="m", messages=_MESSAGES, stream=True))

        self.assertEqual(got, ["a"])
        self.assertEqual(calls["n"], 2)

    def test_sync_midstream_failure_raises_without_retry(self):
        inner = _Transient429("reset mid-stream")
        calls = {"n": 0}

        def fake_completion(**kw):
            calls["n"] += 1
            return _FakeSyncStream(["a", "b"], fail_before_index=1, exc=inner)

        got = []
        with patch.object(litellm, "completion", fake_completion):
            with self.assertRaises(MantisStreamInterruptedError) as cm:
                for chunk in self.client.completion(model="m", messages=_MESSAGES, stream=True):
                    got.append(chunk)

        self.assertEqual(got, ["a"])
        self.assertEqual(cm.exception.chunks_yielded, 1)
        self.assertIs(cm.exception.original_exception, inner)
        self.assertEqual(calls["n"], 1)

    def test_sync_nonstream_path_unchanged(self):
        sentinel = object()
        seen = {}

        def fake_completion(**kw):
            seen.update(kw)
            return sentinel

        with patch.object(litellm, "completion", fake_completion):
            result = self.client.completion(model="m", messages=_MESSAGES)

        self.assertIs(result, sentinel)
        # Sync non-stream has always passed stream explicitly; it must stay False.
        self.assertIs(seen.get("stream"), False)


if __name__ == "__main__":
    unittest.main()
