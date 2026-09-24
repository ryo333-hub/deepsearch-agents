"""Structure-first splitting with exact E5 token checks and source offsets.

Only a local tokenizer is loaded. Encoder weights are never used for chunking.
"""

import hashlib
import re
from bisect import bisect_left, bisect_right

from app.local_rag import config
from app.local_rag.schemas import ChunkRecord, Citation, DocumentRecord, ParsedDocument
from app.local_rag.tokenization import DocumentTokenBudget, get_document_token_budget


def _boundaries(text: str) -> list[list[int]]:
    return [
        [m.end() for m in re.finditer(r"\n[ \t]*\n+", text)],
        [m.end() for m in re.finditer(r"\n", text)],
        [m.end() for m in re.finditer(r"[。！？!?]+[\"'”’）)]*", text)],
        [m.end() for m in re.finditer(r"[.!?]+[\"')]*\s+", text)],
    ]


def _fits(text: str, budget: DocumentTokenBudget) -> bool:
    return not text.strip() or budget.document_tokens(text) <= budget.embedding_max_tokens


def _safe_end(text: str, start: int, cap: int, budget: DocumentTokenBudget) -> int:
    if _fits(text[start:cap], budget):
        return cap
    low, high = start, cap
    # Token count is not strictly monotone across subword merges. Binary search
    # is only a way to find a VERIFIED fitting prefix, not a claim of maximality.
    while low + 1 < high:
        middle = (low + high) // 2
        if _fits(text[start:middle], budget):
            low = middle
        else:
            high = middle
    if low == start:
        raise ValueError("A source character cannot fit the document token budget")
    return low


def _overlap_start(text: str, left: int, right: int, boundaries, budget: DocumentTokenBudget) -> int:
    target = min(config.CHUNK_OVERLAP_TOKENS, budget.overlap_tokens(text[left:right]) // 4)
    if target == 0:
        return right
    low, high = left, right
    # high always denotes an actually checked fitting suffix (empty initially).
    while low + 1 < high:
        middle = (low + high) // 2
        if budget.overlap_tokens(text[middle:right]) <= target:
            high = middle
        else:
            low = middle
    nearby = []
    for group in boundaries:
        position = bisect_left(group, high)
        if position < len(group) and group[position] < right:
            candidate = group[position]
            if budget.overlap_tokens(text[candidate:right]) <= target:
                nearby.append(candidate)
    return min(nearby) if nearby else high


def chunk_document(parsed: ParsedDocument, document_id: str, *,
                   token_budget: DocumentTokenBudget | None = None) -> tuple[ChunkRecord, ...]:
    """token_budget is a trusted backend test seam, never user/model input."""
    maximum = config.MAX_CHUNK_CHARS
    if (not 0 < maximum <= config.MAX_CHUNK_TEXT_LENGTH
            or type(config.CHUNK_OVERLAP_TOKENS) is not int
            or not 0 <= config.CHUNK_OVERLAP_TOKENS < config.EMBEDDING_MAX_INPUT_TOKENS // 2):
        raise ValueError("Invalid chunk token/character budget")
    budget = token_budget if token_budget is not None else get_document_token_budget()
    result = []
    for block in parsed.blocks:
        text = block.text
        boundaries = _boundaries(text)
        newlines = [m.start() for m in re.finditer("\n", text)]
        start, covered_end = 0, 0
        while start < len(text):
            while start < len(text) and text[start].isspace():
                start += 1
            if start == len(text):
                break
            hard_end = _safe_end(text, start, min(start + maximum, len(text)), budget)
            if hard_end <= covered_end:
                # Unusual tokenizer merges may make the overlap consume the
                # available budget. Drop overlap, never source content.
                start = covered_end
                continue
            end = hard_end
            if hard_end < len(text):
                for candidates in boundaries:
                    last = bisect_right(candidates, hard_end) - 1
                    if (last >= 0 and candidates[last] > max(start, covered_end)
                            and _fits(text[start:candidates[last]], budget)):
                        end = candidates[last]
                        break
            # Trim only the chunk's outer whitespace; map back using real offsets.
            piece = text[start:end]
            left = start + len(piece) - len(piece.lstrip())
            right = end - (len(piece) - len(piece.rstrip()))
            emitted = False
            if left < right and right > covered_end:
                value = text[left:right]
                budget.validate_document(value, f"block_{block.block_index}:{left}:{right}")
                first = block.start_line + bisect_left(newlines, left) if block.start_line else None
                last = block.start_line + bisect_left(newlines, right - 1) if block.start_line else None
                result.append(ChunkRecord(
                    document_id=document_id, chunk_index=len(result), text=value,
                    text_hash=hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    page=block.page, start_line=first, end_line=last, heading=block.heading,
                    source_block_index=block.block_index, start_char=left, end_char=right,
                    chunking_version=config.CHUNKING_VERSION,
                ))
                emitted = True
                if len(result) > config.MAX_CHUNKS_PER_DOCUMENT:
                    raise ValueError("文档超过最大 Chunk 数量限制。")
            if end >= len(text):
                break
            covered_end = end
            start = (_overlap_start(text, left, right, boundaries, budget)
                     if emitted else end)
    if not result:
        raise ValueError("文档无可分块文本。")
    return tuple(result)


def citation_from_chunk(chunk: ChunkRecord, document: DocumentRecord,
                        knowledge_base_id: str, citation_id: str = "C1") -> Citation:
    if chunk.document_id != document.document_id:
        raise ValueError("Chunk does not belong to the document")
    return Citation(citation_id=citation_id, document_id=document.document_id,
                    document_name=document.document_name, chunk_id=chunk.chunk_id,
                    knowledge_base_id=knowledge_base_id, page=chunk.page,
                    start_line=chunk.start_line, end_line=chunk.end_line, heading=chunk.heading,
                    source_block_index=chunk.source_block_index,
                    start_char=chunk.start_char, end_char=chunk.end_char,
                    score=None, score_type=None)
