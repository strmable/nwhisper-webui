from typing import List

from modules.safe_mode.chunk_transcriber import ChunkResult
from modules.utils.logger import get_logger

logger = get_logger()


class OffsetCorrector:
    def correct(self, chunk_results: List[ChunkResult]) -> List[ChunkResult]:
        """
        Convert chunk-local timestamps to absolute (original audio) timestamps.
        Modifies segments in-place and returns the same list of ChunkResults
        so that MergeDedup can still access per-chunk membership.

        Parameters
        ----------
        chunk_results : List[ChunkResult]

        Returns
        -------
        List[ChunkResult]
            Same list with absolute timestamps applied to all segments.
        """
        total = sum(len(cr.segments) for cr in chunk_results)

        for cr in chunk_results:
            offset = cr.chunk.audio_start_time
            for seg in cr.segments:
                seg.start = (seg.start or 0.0) + offset
                seg.end = (seg.end or 0.0) + offset

                if seg.words:
                    for word in seg.words:
                        word.start = (word.start or 0.0) + offset
                        word.end = (word.end or 0.0) + offset

                logger.debug(
                    f"[SafeMode] Chunk {cr.chunk.index} seg offset +{offset:.2f}s → "
                    f"[{seg.start:.2f}, {seg.end:.2f}] \"{(seg.text or '').strip()[:40]}\""
                )

        logger.info(
            f"[SafeMode] OffsetCorrector: {total} segments corrected across "
            f"{len(chunk_results)} chunks"
        )
        return chunk_results
