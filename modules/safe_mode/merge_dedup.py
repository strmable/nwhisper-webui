from typing import List

from modules.safe_mode.chunk_transcriber import ChunkResult
from modules.safe_mode.chunk_splitter import Chunk
from modules.whisper.data_classes import Segment
from modules.utils.logger import get_logger

logger = get_logger()


class MergeDedup:
    def merge(
        self,
        chunk_results: List[ChunkResult],
    ) -> List[Segment]:
        """
        Remove duplicate segments at chunk overlap boundaries.

        Because each ChunkResult knows exactly which segments it produced,
        there is no ambiguity in chunk membership.

        Strategy: for each consecutive chunk pair (A, B), the boundary cutoff
        is the midpoint of the overlap region:
            cutoff = (A.audio_end_time + B.audio_start_time) / 2

        From chunk A: keep segments with start < cutoff.
        From chunk B: keep segments with start >= cutoff.
        (Single-chunk input: all segments are kept.)

        Parameters
        ----------
        chunk_results : List[ChunkResult]
            Offset-corrected chunk results (from OffsetCorrector.correct()).

        Returns
        -------
        List[Segment]
            Deduplicated, sorted, and renumbered segments.
        """
        if not chunk_results:
            return []

        if len(chunk_results) == 1:
            return self._renumber(list(chunk_results[0].segments))

        total_before = sum(len(cr.segments) for cr in chunk_results)
        kept: List[Segment] = []

        for i, cr in enumerate(chunk_results):
            chunk = cr.chunk

            # Lower cutoff: boundary with previous chunk
            if i > 0:
                prev_chunk = chunk_results[i - 1].chunk
                low_cutoff = (prev_chunk.audio_end_time + chunk.audio_start_time) / 2.0
            else:
                low_cutoff = 0.0

            # Upper cutoff: boundary with next chunk
            if i < len(chunk_results) - 1:
                next_chunk = chunk_results[i + 1].chunk
                high_cutoff = (chunk.audio_end_time + next_chunk.audio_start_time) / 2.0
            else:
                high_cutoff = float("inf")

            for seg in cr.segments:
                start = seg.start or 0.0
                if start < low_cutoff:
                    logger.debug(
                        f"[SafeMode] Drop seg [{start:.2f}s] from chunk {i} "
                        f"(below low_cutoff {low_cutoff:.2f}s): \"{(seg.text or '').strip()[:40]}\""
                    )
                    continue
                if start >= high_cutoff:
                    logger.debug(
                        f"[SafeMode] Drop seg [{start:.2f}s] from chunk {i} "
                        f"(above high_cutoff {high_cutoff:.2f}s): \"{(seg.text or '').strip()[:40]}\""
                    )
                    continue
                kept.append(seg)

        kept.sort(key=lambda s: s.start or 0.0)

        logger.info(
            f"[SafeMode] MergeDedup: {total_before} → {len(kept)} segments "
            f"({total_before - len(kept)} removed)"
        )

        return self._renumber(kept)

    @staticmethod
    def _renumber(segments: List[Segment]) -> List[Segment]:
        for i, seg in enumerate(segments, start=1):
            seg.id = i
        return segments
