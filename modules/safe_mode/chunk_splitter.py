import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from modules.utils.logger import get_logger

logger = get_logger()

SAMPLE_RATE = 16000
RMS_WINDOW_SEC = 0.1  # window size for RMS energy computation


@dataclass
class Chunk:
    index: int
    audio_start_time: float   # actual audio start in seconds (includes overlap with prev)
    audio_end_time: float     # actual audio end in seconds (includes overlap with next)
    content_start_time: float # canonical split-point start
    content_end_time: float   # canonical split-point end
    audio_segment: np.ndarray # audio data slice


class ChunkSplitter:
    def split(
        self,
        speech_chunks: List[Dict],
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        max_chunk_sec: float = 30.0,
        overlap_sec: float = 1.0,
    ) -> List[Chunk]:
        """
        Split audio into chunks based on VAD speech_chunks.

        Parameters
        ----------
        speech_chunks : List[Dict]
            VAD output: [{"start": int_samples, "end": int_samples}, ...]
        audio : np.ndarray
            Original audio array (1-D, float32, 16kHz)
        sample_rate : int
        max_chunk_sec : float
        overlap_sec : float

        Returns
        -------
        List[Chunk]
        """
        total_samples = len(audio)
        total_duration = total_samples / sample_rate

        if speech_chunks:
            speech_segs = [(c["start"] / sample_rate, c["end"] / sample_rate)
                           for c in speech_chunks]
        else:
            speech_segs = [(0.0, total_duration)]

        logger.debug(
            f"[SafeMode] ChunkSplitter: {len(speech_segs)} VAD segments, "
            f"total={total_duration:.1f}s, max_chunk={max_chunk_sec}s, overlap={overlap_sec}s"
        )

        split_points = self._find_split_points(speech_segs, audio, sample_rate, max_chunk_sec)

        logger.info(
            f"[SafeMode] Creating {len(split_points)} chunks from "
            f"{total_duration:.1f}s audio"
        )

        chunks = []
        for i, (cs, ce) in enumerate(split_points):
            audio_start = max(0.0, cs - overlap_sec)
            audio_end = min(total_duration, ce + overlap_sec)

            start_sample = int(audio_start * sample_rate)
            end_sample = min(int(audio_end * sample_rate), total_samples)
            audio_segment = audio[start_sample:end_sample].copy()

            chunk = Chunk(
                index=i,
                audio_start_time=audio_start,
                audio_end_time=audio_end,
                content_start_time=cs,
                content_end_time=ce,
                audio_segment=audio_segment,
            )
            chunks.append(chunk)
            logger.debug(
                f"[SafeMode] Chunk {i}: audio=[{audio_start:.2f}, {audio_end:.2f}]s "
                f"content=[{cs:.2f}, {ce:.2f}]s len={len(audio_segment)/sample_rate:.2f}s"
            )

        return chunks

    def _find_split_points(
        self,
        speech_segs: List[Tuple[float, float]],
        audio: np.ndarray,
        sample_rate: int,
        max_chunk_sec: float,
    ) -> List[Tuple[float, float]]:
        """Group speech segments into chunks, splitting at silence gaps or low-RMS points."""
        total_duration = len(audio) / sample_rate
        result: List[Tuple[float, float]] = []
        current_group: List[Tuple[float, float]] = []
        current_duration = 0.0

        for i, (seg_start, seg_end) in enumerate(speech_segs):
            seg_dur = seg_end - seg_start

            # A single segment exceeds max_chunk_sec → split by RMS
            if seg_dur > max_chunk_sec:
                # Flush accumulated group first
                if current_group:
                    result.append((current_group[0][0], current_group[-1][1]))
                    current_group = []
                    current_duration = 0.0

                # RMS-based sub-split of this long segment
                sub_points = self._rms_split(audio, sample_rate, seg_start, seg_end, max_chunk_sec)
                result.extend(sub_points)
                continue

            # Adding this segment would exceed max_chunk_sec → flush current group first
            if current_duration + seg_dur > max_chunk_sec and current_group:
                # Split at the midpoint of the silence gap between last group segment and this one
                prev_end = current_group[-1][1]
                gap_mid = (prev_end + seg_start) / 2.0
                result.append((current_group[0][0], gap_mid))
                current_group = [(seg_start, seg_end)]
                current_duration = seg_dur
            else:
                current_group.append((seg_start, seg_end))
                current_duration += seg_dur

        # Flush remaining group
        if current_group:
            result.append((current_group[0][0], total_duration))
        elif not result:
            result.append((0.0, total_duration))

        return result

    def _rms_split(
        self,
        audio: np.ndarray,
        sample_rate: int,
        t_start: float,
        t_end: float,
        max_chunk_sec: float,
    ) -> List[Tuple[float, float]]:
        """
        Split a long speech segment using RMS energy minima.
        Recursively splits until all sub-segments are <= max_chunk_sec.
        """
        duration = t_end - t_start
        if duration <= max_chunk_sec:
            return [(t_start, t_end)]

        split_t = self._find_rms_split_point(audio, sample_rate, t_start, t_end)
        logger.debug(
            f"[SafeMode] RMS split [{t_start:.2f}, {t_end:.2f}]s → split at {split_t:.2f}s"
        )

        left = self._rms_split(audio, sample_rate, t_start, split_t, max_chunk_sec)
        right = self._rms_split(audio, sample_rate, split_t, t_end, max_chunk_sec)
        return left + right

    def _find_rms_split_point(
        self,
        audio: np.ndarray,
        sample_rate: int,
        t_start: float,
        t_end: float,
    ) -> float:
        """
        Find the time point of minimum RMS energy within [t_start, t_end].
        Returns a float in seconds (absolute, not relative).
        """
        start_sample = int(t_start * sample_rate)
        end_sample = min(int(t_end * sample_rate), len(audio))
        segment = audio[start_sample:end_sample]

        window_samples = max(1, int(RMS_WINDOW_SEC * sample_rate))
        n_windows = len(segment) // window_samples

        if n_windows < 2:
            # Segment too short to find a meaningful split, return midpoint
            return (t_start + t_end) / 2.0

        rms_values = np.array([
            np.sqrt(np.mean(segment[j * window_samples:(j + 1) * window_samples] ** 2))
            for j in range(n_windows)
        ])

        # Find minimum RMS in the middle half of the segment to avoid boundary effects
        quarter = n_windows // 4
        search_range = rms_values[quarter: n_windows - quarter]
        if len(search_range) == 0:
            search_range = rms_values

        min_idx = quarter + int(np.argmin(search_range))
        split_sample = start_sample + min_idx * window_samples
        return split_sample / sample_rate
