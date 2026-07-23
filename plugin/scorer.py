"""BGE-M3 scoring engine for PACE predictive context compression.

Encodes the current query (user message + recent N rounds) and scores it
against cached chunk key embeddings using cosine similarity. The resulting
attention weights drive the 4-level granularity selection in the compressor.

PACE Algorithm 1 steps 1, 4-6:
  1. q_t ← BGE-M3-Encode(Q ⊕ R_full^{(t-N:t-1)})
  4. s_i ← cosine(q_t, k_i) for all i
  5. w_i ← softmax(s_i / τ)
  6. w̃_i ← M · w_i

References:
  - PACE design §5.1 Algorithm 1
  - BGE-M3: Chen et al. 2024 (1024-dim multilingual embeddings)
  - Degradation: BGE-M3 unavailable → all R_full (design §6.4)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class PACEScorer:
    """BGE-M3 based relevance scorer for PACE compression.

    Loads the BGE-M3 sentence-transformers model on first use and caches
    chunk key embeddings. All inference runs on CPU (target <1s/step).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model: Any = None
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Return True when the BGE-M3 model is available for inference."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Last error encountered during model loading, if any."""
        return self._load_error

    def load_model(self) -> bool:
        """Load the BGE-M3 model. Returns True on success.

        On failure, sets ``self._load_error`` and returns False. The
        compressor will degrade to all-R_full mode when the scorer is
        unavailable.
        """
        if self._model is not None:
            return True
        try:
            # Windows GFW workaround: force hf-mirror.com for model download
            import os as _os
            _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer

            logger.info(
                "PACE scorer: loading BGE-M3 model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._load_error = None
            logger.info("PACE scorer: BGE-M3 loaded successfully (dim=%d).", self._model.get_sentence_embedding_dimension())
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None
            logger.warning("PACE scorer: BGE-M3 load failed: %s", exc)
            return False

    def _extract_text(self, messages: Sequence[dict]) -> str:
        """Flatten a message list into a single text string."""
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
        return "\n".join(parts)

    def encode_query(
        self,
        user_message: str,
        recent_2_full: Sequence[dict],
    ) -> Any | None:
        """Encode the query: Q ⊕ R_full^{(t-N:t-1)}.

        PACE step 1: concatenate the current user message with the last
        N=2 rounds of full conversation history, then BGE-M3-encode.

        Returns the query embedding (numpy array, 1024-dim) or None if
        the model is unavailable.
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        recent_text = self._extract_text(recent_2_full)
        query_text = user_message + "\n" + recent_text if recent_text else user_message

        try:
            embedding = self._model.encode(
                query_text,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            return embedding
        except Exception as exc:
            logger.warning("PACE scorer: encode_query failed: %s", exc)
            return None

    def encode_key(self, chunk_text: str) -> Any | None:
        """Encode a chunk's key text for caching in the memory store.

        PACE step 13: k_t ← BGE-M3-Encode(Trunc(R_full^{(t)}, L_max)).
        The caller is responsible for truncation; this method just encodes.
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        try:
            embedding = self._model.encode(
                chunk_text,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            return embedding
        except Exception as exc:
            logger.warning("PACE scorer: encode_key failed: %s", exc)
            return None

    def score(
        self,
        query_embedding: Any,
        cached_keys: Sequence[Any],
    ) -> Any:
        """Compute cosine similarity scores (PACE steps 4-6).

        With normalized embeddings, cosine similarity = dot product.

        Returns a numpy array of scores. If ``cached_keys`` is empty,
        returns an empty array. If the model is unavailable or inputs
        are invalid, returns an empty array (all chunks default to R_ph).

        PACE step 4: s_i ← cosine(q_t, k_i)
        PACE step 5: w_i ← softmax(s_i / τ)  — done in compressor
        PACE step 6: w̃_i ← M · w_i           — done in compressor
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("PACE scorer: numpy not available")
            return _empty_array()

        if query_embedding is None or len(cached_keys) == 0:
            return _empty_array()

        try:
            keys_matrix = np.array(cached_keys, dtype=np.float32)
            q = np.array(query_embedding, dtype=np.float32)

            # cosine similarity = dot product (embeddings are normalized)
            scores = keys_matrix @ q
            return scores
        except Exception as exc:
            logger.warning("PACE scorer: score computation failed: %s", exc)
            return _empty_array()


def _empty_array():
    """Return an empty numpy array (lazy import to avoid hard dependency)."""
    try:
        import numpy as np
        return np.array([], dtype=np.float32)
    except ImportError:
        return []
