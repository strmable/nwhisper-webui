from typing import List, Optional

from modules.safe_mode.chunk_transcriber import ChunkResult
from modules.safe_mode.merge_dedup import MergeDedup
from modules.whisper.data_classes import Segment
from modules.utils.logger import get_logger

logger = get_logger()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.debug("[SafeMode] google-generativeai not installed; GeminiMerger will fall back to MergeDedup")


class GeminiMerger:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model = model

    def merge(
        self,
        chunk_results: List[ChunkResult],
        context_sentences: int = 3,
    ) -> List[Segment]:
        """
        Use Gemini LLM to refine overlap boundaries between chunks.

        Falls back to MergeDedup if:
        - google-generativeai is not installed
        - API key is missing
        - Any API call fails

        Parameters
        ----------
        chunk_results : List[ChunkResult]
            Offset-corrected chunk results.
        context_sentences : int
            Number of sentences from each side to include as context.

        Returns
        -------
        List[Segment]
        """
        if not GEMINI_AVAILABLE or not self.api_key:
            logger.info("[SafeMode] GeminiMerger: falling back to MergeDedup "
                        f"(available={GEMINI_AVAILABLE}, key_set={bool(self.api_key)})")
            return MergeDedup().merge(chunk_results)

        # First apply rule-based dedup to get a clean baseline
        deduped = MergeDedup().merge(chunk_results)
        chunks = [cr.chunk for cr in chunk_results]

        try:
            genai.configure(api_key=self.api_key)
            model_client = genai.GenerativeModel(self.model)
        except Exception as e:
            logger.warning(f"[SafeMode] GeminiMerger init failed: {e}. Falling back to MergeDedup result.")
            return deduped

        # Refine text at each chunk boundary
        for i in range(len(chunks) - 1):
            cutoff = (chunks[i].audio_end_time + chunks[i + 1].audio_start_time) / 2.0

            # Gather context segments around the boundary
            before = [s for s in deduped if (s.start or 0) < cutoff][-context_sentences:]
            after = [s for s in deduped if (s.start or 0) >= cutoff][:context_sentences]

            if not before or not after:
                continue

            before_text = " ".join(s.text or "" for s in before).strip()
            after_text = " ".join(s.text or "" for s in after).strip()

            prompt = (
                "The following are two consecutive transcription segments from a speech audio file. "
                "They were transcribed independently and may have a slightly awkward boundary. "
                "Please return ONLY the corrected versions of these two text blocks, "
                "separated by the delimiter '|||'. "
                "Do not add any explanation.\n\n"
                f"BEFORE BOUNDARY:\n{before_text}\n\n"
                f"AFTER BOUNDARY:\n{after_text}"
            )

            try:
                response = model_client.generate_content(prompt)
                result_text = response.text.strip()
                parts = result_text.split("|||")
                if len(parts) == 2:
                    new_before, new_after = parts[0].strip(), parts[1].strip()
                    # Distribute corrected text back to segments (simple proportional split)
                    self._redistribute_text(before, new_before)
                    self._redistribute_text(after, new_after)
                    logger.debug(
                        f"[SafeMode] GeminiMerger: boundary {i}↔{i+1} at {cutoff:.2f}s refined"
                    )
                else:
                    logger.debug(
                        f"[SafeMode] GeminiMerger: unexpected response format at boundary {i}, skipping"
                    )
            except Exception as e:
                logger.warning(
                    f"[SafeMode] GeminiMerger: API error at boundary {i}: {e}. Keeping MergeDedup result."
                )

        logger.info(f"[SafeMode] GeminiMerger: finished processing {len(chunks) - 1} boundaries")
        return deduped

    @staticmethod
    def _redistribute_text(segments: List[Segment], new_text: str) -> None:
        """
        Distribute corrected text back to segments proportionally by original text length.
        Timestamps are preserved; only text is updated.
        """
        if not segments:
            return

        original_lengths = [len(s.text or "") for s in segments]
        total_orig = sum(original_lengths) or 1
        words = new_text.split()
        total_words = len(words)
        cursor = 0

        for j, seg in enumerate(segments):
            share = original_lengths[j] / total_orig
            n = round(share * total_words)
            if j == len(segments) - 1:
                seg.text = " ".join(words[cursor:])
            else:
                seg.text = " ".join(words[cursor:cursor + n])
                cursor += n
