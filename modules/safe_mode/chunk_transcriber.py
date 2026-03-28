import gradio as gr
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from copy import deepcopy

from modules.safe_mode.chunk_splitter import Chunk
from modules.whisper.data_classes import Segment, WhisperParams
from modules.utils.logger import get_logger

logger = get_logger()


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

        logger.info(
            f"[SafeMode] All {total} chunks transcribed. "
            f"Total raw segments: {sum(len(r.segments) for r in results)}"
        )
        return results
