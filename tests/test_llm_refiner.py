"""API is mocked: no subtitle text leaves the machine during tests."""

import json
import unittest

import requests

from core.llm_refiner import LLMConfig, refine_subtitles
from core.srt_builder import SubtitleEntry


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


class _Session:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple] = []

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(next(self.responses))


class _FailingSession:
    def post(self, *args, **kwargs):
        raise requests.Timeout("request timed out")


class LLMRefinerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = [SubtitleEntry(1, 100, 900, "Xin chao"),
                        SubtitleEntry(2, 1000, 1800, "Tam biet")]

    def test_disabled_preserves_entries_without_request(self) -> None:
        session = _Session([])
        self.assertEqual(refine_subtitles(self.entries, session=session), self.entries)
        self.assertEqual(session.calls, [])

    def test_valid_response_changes_only_text(self) -> None:
        session = _Session([json.dumps({"1": "Xin chào", "2": "Tạm biệt"}, ensure_ascii=False)])
        config = LLMConfig(True, "http://localhost:11434/v1", None, "local-model")
        result = refine_subtitles(self.entries, config, session=session)
        self.assertEqual([item.text for item in result], ["Xin chào", "Tạm biệt"])
        self.assertEqual([(item.id, item.start_ms, item.end_ms) for item in result],
                         [(item.id, item.start_ms, item.end_ms) for item in self.entries])
        self.assertEqual(session.calls[0][0], "http://localhost:11434/v1/chat/completions")

    def test_missing_id_keeps_original_batch(self) -> None:
        session = _Session(['{"1":"Xin chào"}'])
        config = LLMConfig(True, "http://localhost:11434/v1", None, "local-model")
        self.assertEqual(refine_subtitles(self.entries, config, session=session), self.entries)

    def test_batches_are_independent(self) -> None:
        session = _Session(['{"1":"Xin chào"}', '{"unexpected":"text"}'])
        config = LLMConfig(True, "http://localhost:11434/v1", None, "local-model", batch_size=1)
        result = refine_subtitles(self.entries, config, session=session)
        self.assertEqual([item.text for item in result], ["Xin chào", "Tam biet"])

    def test_obvious_translation_is_rejected(self) -> None:
        originals = [SubtitleEntry(1, 0, 1000, "你好世界")]
        session = _Session(['{"1":"Hello world"}'])
        config = LLMConfig(True, "http://localhost:11434/v1", None, "local-model")
        self.assertEqual(refine_subtitles(originals, config, session=session), originals)

    def test_network_timeout_keeps_originals(self) -> None:
        config = LLMConfig(True, "http://localhost:11434/v1", None, "local-model")
        self.assertEqual(refine_subtitles(self.entries, config, session=_FailingSession()), self.entries)


if __name__ == "__main__":
    unittest.main()
