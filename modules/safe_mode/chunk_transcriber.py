import os
import datetime
import gradio as gr
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from copy import deepcopy

from modules.safe_mode.chunk_splitter import Chunk
from modules.whisper.data_classes import Segment, WhisperParams
from modules.utils.logger import get_logger
from modules.utils.paths import OUTPUT_DIR

logger = get_logger()


def _seconds_to_srt_time(seconds: float) -> str:
    td = datetime.timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    millis = int((td.total_seconds() - total_seconds) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _save_partial_srt(segments: List[Segment], checkpoint_path: str) -> None:
    """Write accumulated segments to SRT (overwrites on each call)."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = _seconds_to_srt_time(seg.start or 0.0)
            end = _seconds_to_srt_time(seg.end or 0.0)
            text = (seg.text or "").strip()
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")


@dataclass
class ChunkResult:
    chunk: Chunk
    segments: List[Segment] = field(default_factory=list)


class _ScaledProgress:
    """
    Wraps gr.Progress so that a chunk-local [0, 1] value maps to
    the chunk's slice [lo, hi] of the overall [0, 1] progress bar.
    This prevents the bar from jumping back to 0 at the start of each chunk.
    """
    def __init__(self, base: gr.Progress, chunk_idx: int, total: int):
        self._base = base
        self._lo = chunk_idx / total
        self._hi = (chunk_idx + 1) / total

    def __call__(self, value=0, desc=None, **kwargs):
        scaled = self._lo + float(value) * (self._hi - self._lo)
        self._base(scaled, desc=desc, **kwargs)

    def tqdm(self, *args, **kwargs):
        return self._base.tqdm(*args, **kwargs)


class ChunkTranscriber:
    def transcribe_chunks(
        self,
        chunks: List[Chunk],
        pipeline,
        whisper_params: WhisperParams,
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        source_name: str = "safe_mode",
    ) -> List[ChunkResult]:
        """
        Transcribe each chunk independently.

        Parameters
        ----------
        chunks : List[Chunk]
        pipeline : BaseTranscriptionPipeline
            The whisper pipeline instance with a transcribe() method.
        whisper_params : WhisperParams
        progress : gr.Progress
        progress_callback : Optional[Callable]

        Returns
        -------
        List[ChunkResult]
            Each result contains the chunk and its locally-timestamped segments.
        """
        # Force condition_on_previous_text=False for each chunk
        chunk_params = deepcopy(whisper_params)
        chunk_params.condition_on_previous_text = False
        params_list = chunk_params.to_list()

        total = len(chunks)
        results = []

        # Checkpoint file: overwritten after every chunk so partial results
        # survive if an error occurs mid-way.
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
        checkpoint_path = os.path.join(OUTPUT_DIR, f"{safe_name}_checkpoint.srt")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"[SafeMode] Checkpoint file: {checkpoint_path}")

        for i, chunk in enumerate(chunks):
            progress_ratio = i / total
            desc = f"[SafeMode] Transcribing chunk {i + 1}/{total} " \
                   f"({chunk.content_start_time:.1f}~{chunk.content_end_time:.1f}s).."
            progress(progress_ratio, desc=desc)
            if progress_callback is not None:
                progress_callback(progress_ratio)

            logger.debug(
                f"[SafeMode] Chunk {i + 1}/{total}: "
                f"audio=[{chunk.audio_start_time:.2f}, {chunk.audio_end_time:.2f}]s, "
                f"audio_len={len(chunk.audio_segment)} samples"
            )

            scaled_progress = _ScaledProgress(progress, i, total)
            segments, elapsed = pipeline.transcribe(
                chunk.audio_segment,
                scaled_progress,
                None,  # suppress inner progress_callback; we manage it here
                *params_list,
            )

            logger.debug(
                f"[SafeMode] Chunk {i + 1}/{total}: got {len(segments)} segments "
                f"in {elapsed:.2f}s"
            )

            results.append(ChunkResult(chunk=chunk, segments=segments))

            # Save partial SRT after every chunk.
            # Make a shallow copy so the checkpoint offset does not affect the
            # main results list (segments are deepcopied per chunk).
            from modules.safe_mode.offset_corrector import OffsetCorrector
            from copy import deepcopy as _deepcopy
            partial_copy = [ChunkResult(chunk=cr.chunk, segments=_deepcopy(cr.segments)) for cr in results]
            partial_chunk_results = OffsetCorrector().correct(partial_copy)
            partial_segments = [s for cr in partial_chunk_results for s in cr.segments]
            partial_segments.sort(key=lambda s: s.start or 0.0)
            _save_partial_srt(partial_segments, checkpoint_path)
            logger.debug(
                f"[SafeMode] Checkpoint saved: {len(partial_segments)} segments → {checkpoint_path}"
            )

        logger.info(
            f"[SafeMode] All {total} chunks transcribed. "
            f"Total raw segments: {sum(len(r.segments) for r in results)}"
        )
        return results
