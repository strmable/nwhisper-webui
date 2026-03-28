"""
Unit tests for Long File Safe Mode modules.

Mock pipeline is used so no actual Whisper model is required.
"""
import numpy as np
import pytest
import gradio as gr
from typing import List, Tuple, Union, Callable, Optional, BinaryIO
from copy import deepcopy

from modules.safe_mode.chunk_splitter import ChunkSplitter, Chunk
from modules.safe_mode.chunk_transcriber import ChunkTranscriber, ChunkResult
from modules.safe_mode.offset_corrector import OffsetCorrector
from modules.safe_mode.merge_dedup import MergeDedup

# MergeDedup now takes List[ChunkResult], not (List[Segment], List[Chunk])
from modules.whisper.data_classes import (
    Segment, Word, SafeModeParams, WhisperParams, TranscriptionPipelineParams
)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_audio(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Create a synthetic audio signal (sine wave)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def make_silent_audio(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def make_speech_chunks(intervals: List[Tuple[float, float]], sample_rate: int = SAMPLE_RATE) -> List[dict]:
    """Create VAD-style speech_chunks from (start_sec, end_sec) pairs."""
    return [{"start": int(s * sample_rate), "end": int(e * sample_rate)} for s, e in intervals]


def make_segment(start: float, end: float, text: str, seg_id: int = 1) -> Segment:
    return Segment(id=seg_id, start=start, end=end, text=text)


class MockPipeline:
    """
    Minimal mock that mimics BaseTranscriptionPipeline.transcribe().
    Returns fake segments whose timestamps start at 0 (local time).
    """
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        # Derive local duration from audio length
        if isinstance(audio, np.ndarray):
            duration = len(audio) / SAMPLE_RATE
        else:
            duration = 10.0  # fallback

        # Emit two fake segments covering the whole chunk at local time
        segments = [
            Segment(id=1, start=0.0, end=duration * 0.5, text=" hello world"),
            Segment(id=2, start=duration * 0.5, end=duration, text=" foo bar"),
        ]
        return segments, 0.1


# ---------------------------------------------------------------------------
# ChunkSplitter tests
# ---------------------------------------------------------------------------

class TestChunkSplitter:
    def test_single_chunk_short_audio(self):
        """Audio shorter than max_chunk_sec produces exactly 1 chunk."""
        audio = make_audio(10.0)
        speech_chunks = make_speech_chunks([(0.5, 9.5)])
        chunks = ChunkSplitter().split(speech_chunks, audio, max_chunk_sec=30.0, overlap_sec=1.0)
        assert len(chunks) == 1
        assert chunks[0].content_start_time == pytest.approx(0.5, abs=0.1)

    def test_two_chunks_from_long_audio(self):
        """Audio with two speech regions separated by silence → 2 chunks."""
        audio = make_audio(80.0)
        speech_chunks = make_speech_chunks([(0.0, 35.0), (45.0, 75.0)])
        chunks = ChunkSplitter().split(speech_chunks, audio, max_chunk_sec=40.0, overlap_sec=1.0)
        assert len(chunks) == 2

    def test_overlap_extends_audio_segment(self):
        """Each chunk's audio segment should extend overlap_sec beyond content bounds."""
        audio = make_audio(60.0)
        speech_chunks = make_speech_chunks([(0.0, 25.0), (35.0, 55.0)])
        overlap = 1.5
        chunks = ChunkSplitter().split(speech_chunks, audio, max_chunk_sec=30.0, overlap_sec=overlap)
        for chunk in chunks:
            # audio range must be at least as wide as content range
            assert chunk.audio_start_time <= chunk.content_start_time
            assert chunk.audio_end_time >= chunk.content_end_time
            # audio_start should be content_start - overlap (clamped to 0)
            expected_start = max(0.0, chunk.content_start_time - overlap)
            assert chunk.audio_start_time == pytest.approx(expected_start, abs=0.01)

    def test_rms_split_for_long_single_segment(self):
        """A single VAD segment exceeding max_chunk_sec triggers RMS-based splitting."""
        duration = 90.0
        audio = make_audio(duration)
        speech_chunks = make_speech_chunks([(0.0, duration)])
        chunks = ChunkSplitter().split(speech_chunks, audio, max_chunk_sec=30.0, overlap_sec=1.0)
        assert len(chunks) >= 3  # 90s / 30s = 3 chunks minimum

    def test_audio_segment_samples_match_time(self):
        """Each chunk's audio_segment length matches (audio_end - audio_start) * sample_rate."""
        audio = make_audio(60.0)
        speech_chunks = make_speech_chunks([(0.0, 25.0), (35.0, 55.0)])
        chunks = ChunkSplitter().split(speech_chunks, audio, max_chunk_sec=30.0, overlap_sec=1.0)
        for chunk in chunks:
            expected_samples = int((chunk.audio_end_time - chunk.audio_start_time) * SAMPLE_RATE)
            assert abs(len(chunk.audio_segment) - expected_samples) <= 1  # rounding tolerance

    def test_no_speech_chunks_fallback(self):
        """Empty speech_chunks → treat whole audio as one speech region."""
        audio = make_audio(20.0)
        chunks = ChunkSplitter().split([], audio, max_chunk_sec=30.0, overlap_sec=1.0)
        assert len(chunks) == 1
        assert chunks[0].audio_start_time == 0.0


# ---------------------------------------------------------------------------
# ChunkTranscriber tests
# ---------------------------------------------------------------------------

class TestChunkTranscriber:
    def _make_chunks(self, n: int = 3, chunk_dur: float = 10.0) -> List[Chunk]:
        chunks = []
        for i in range(n):
            start = i * chunk_dur
            end = start + chunk_dur
            audio = make_audio(chunk_dur)
            chunks.append(Chunk(
                index=i,
                audio_start_time=start,
                audio_end_time=end,
                content_start_time=start,
                content_end_time=end,
                audio_segment=audio,
            ))
        return chunks

    def test_returns_one_result_per_chunk(self):
        chunks = self._make_chunks(3)
        pipeline = MockPipeline()
        params = WhisperParams()
        results = ChunkTranscriber().transcribe_chunks(chunks, pipeline, params)
        assert len(results) == 3

    def test_condition_on_previous_text_forced_false(self):
        """Ensure condition_on_previous_text is False regardless of input params."""
        chunks = self._make_chunks(2)
        pipeline = MockPipeline()
        params = WhisperParams(condition_on_previous_text=True)

        received_params = []
        original_transcribe = pipeline.transcribe

        def recording_transcribe(audio, progress=gr.Progress(), progress_callback=None, *wp):
            p = WhisperParams.from_list(list(wp))
            received_params.append(p.condition_on_previous_text)
            return original_transcribe(audio, progress, progress_callback, *wp)

        pipeline.transcribe = recording_transcribe
        ChunkTranscriber().transcribe_chunks(chunks, pipeline, params)

        assert all(v is False for v in received_params), \
            f"Expected all False, got {received_params}"

    def test_segments_have_local_timestamps(self):
        """Segments from mock should start at local time (close to 0)."""
        chunks = self._make_chunks(2, chunk_dur=10.0)
        pipeline = MockPipeline()
        results = ChunkTranscriber().transcribe_chunks(chunks, pipeline, WhisperParams())
        for r in results:
            assert r.segments[0].start == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# OffsetCorrector tests
# ---------------------------------------------------------------------------

class TestOffsetCorrector:
    def _make_chunk_result(self, chunk_index: int, offset: float, n_segs: int = 2) -> ChunkResult:
        dur = 10.0
        chunk = Chunk(
            index=chunk_index,
            audio_start_time=offset,
            audio_end_time=offset + dur,
            content_start_time=offset,
            content_end_time=offset + dur,
            audio_segment=make_audio(dur),
        )
        segments = [
            Segment(id=j + 1, start=float(j * 2), end=float(j * 2 + 1), text=f"seg{j}")
            for j in range(n_segs)
        ]
        return ChunkResult(chunk=chunk, segments=segments)

    def test_offset_applied_correctly(self):
        results = [
            self._make_chunk_result(0, offset=0.0),
            self._make_chunk_result(1, offset=10.0),
        ]
        corrected_results = OffsetCorrector().correct(results)
        # Chunk 0 starts at 0.0 — no shift
        assert corrected_results[0].segments[0].start == pytest.approx(0.0, abs=0.01)
        assert corrected_results[0].segments[1].start == pytest.approx(2.0, abs=0.01)
        # Chunk 1 offset = 10.0 — all segments shifted by 10
        assert corrected_results[1].segments[0].start == pytest.approx(10.0, abs=0.01)
        assert corrected_results[1].segments[1].start == pytest.approx(12.0, abs=0.01)

    def test_word_timestamps_also_corrected(self):
        offset = 5.0
        chunk = Chunk(
            index=0,
            audio_start_time=offset,
            audio_end_time=offset + 10.0,
            content_start_time=offset,
            content_end_time=offset + 10.0,
            audio_segment=make_audio(10.0),
        )
        word = Word(start=1.0, end=2.0, word="hello", probability=0.99)
        seg = Segment(id=1, start=1.0, end=2.0, text="hello", words=[word])
        result = ChunkResult(chunk=chunk, segments=[seg])

        corrected_results = OffsetCorrector().correct([result])
        seg = corrected_results[0].segments[0]
        assert seg.start == pytest.approx(offset + 1.0, abs=0.01)
        assert seg.words[0].start == pytest.approx(offset + 1.0, abs=0.01)
        assert seg.words[0].end == pytest.approx(offset + 2.0, abs=0.01)


# ---------------------------------------------------------------------------
# MergeDedup tests
# ---------------------------------------------------------------------------

class TestMergeDedup:
    def _make_cr(self, index: int, audio_start: float, audio_end: float,
                 content_start: float, content_end: float,
                 segs: List[Segment]) -> ChunkResult:
        dur = audio_end - audio_start
        chunk = Chunk(
            index=index,
            audio_start_time=audio_start,
            audio_end_time=audio_end,
            content_start_time=content_start,
            content_end_time=content_end,
            audio_segment=make_audio(dur),
        )
        return ChunkResult(chunk=chunk, segments=segs)

    def test_single_chunk_no_dedup(self):
        cr = self._make_cr(0, 0.0, 31.0, 0.0, 30.0, [
            make_segment(1.0, 5.0, "hello", 1),
            make_segment(10.0, 15.0, "world", 2),
        ])
        result = MergeDedup().merge([cr])
        assert len(result) == 2

    def test_duplicate_removed_at_boundary(self):
        """
        Chunk A: audio=[0, 31], content=[0, 30]
        Chunk B: audio=[29, 60], content=[30, 60]
        cutoff = (31 + 29) / 2 = 30.0

        Segment from B at start=29.5 < cutoff → dropped.
        """
        cr_a = self._make_cr(0, 0.0, 31.0, 0.0, 30.0, [
            make_segment(5.0, 10.0, "from A early", 1),
            make_segment(28.0, 30.5, "from A overlap", 2),
        ])
        cr_b = self._make_cr(1, 29.0, 60.0, 30.0, 60.0, [
            make_segment(29.5, 31.0, "from B overlap", 3),  # start < cutoff 30.0 → dropped
            make_segment(35.0, 40.0, "from B main", 4),
        ])
        result = MergeDedup().merge([cr_a, cr_b])

        texts = [s.text.strip() for s in result]
        assert "from A early" in texts
        assert "from A overlap" in texts
        assert "from B main" in texts
        assert "from B overlap" not in texts

    def test_segments_sorted_by_start(self):
        cr_a = self._make_cr(0, 0.0, 31.0, 0.0, 30.0, [
            make_segment(5.0, 7.0, "early A", 1),
            make_segment(15.0, 17.0, "mid A", 2),
        ])
        cr_b = self._make_cr(1, 29.0, 60.0, 30.0, 60.0, [
            make_segment(32.0, 34.0, "main B", 3),
            make_segment(40.0, 42.0, "late B", 4),
        ])
        result = MergeDedup().merge([cr_a, cr_b])
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_ids_renumbered_sequentially(self):
        cr_a = self._make_cr(0, 0.0, 31.0, 0.0, 30.0, [make_segment(5.0, 7.0, "A", 99)])
        cr_b = self._make_cr(1, 29.0, 60.0, 30.0, 60.0, [make_segment(35.0, 37.0, "B", 42)])
        result = MergeDedup().merge([cr_a, cr_b])
        assert [s.id for s in result] == list(range(1, len(result) + 1))


# ---------------------------------------------------------------------------
# TranscriptionPipelineParams round-trip test
# ---------------------------------------------------------------------------

class TestTranscriptionPipelineParams:
    def test_to_list_from_list_roundtrip(self):
        """to_list() then from_list() must reproduce the same params."""
        p = TranscriptionPipelineParams(
            safe_mode=SafeModeParams(
                enabled=True,
                max_chunk_length_sec=25.0,
                chunk_overlap_sec=2.0,
                merge_dedup=False,
                gemini_enabled=True,
                gemini_api_key="test-key",
                gemini_model="gemini-2.0-flash",
                gemini_context_sentences=5,
            )
        )
        lst = p.to_list()
        p2 = TranscriptionPipelineParams.from_list(lst)

        assert p2.safe_mode.enabled is True
        assert p2.safe_mode.max_chunk_length_sec == pytest.approx(25.0)
        assert p2.safe_mode.chunk_overlap_sec == pytest.approx(2.0)
        assert p2.safe_mode.merge_dedup is False
        assert p2.safe_mode.gemini_enabled is True
        assert p2.safe_mode.gemini_api_key == "test-key"
        assert p2.safe_mode.gemini_model == "gemini-2.0-flash"
        assert p2.safe_mode.gemini_context_sentences == 5

    def test_default_safe_mode_disabled(self):
        """Default TranscriptionPipelineParams has safe_mode.enabled=False."""
        p = TranscriptionPipelineParams()
        assert p.safe_mode.enabled is False
