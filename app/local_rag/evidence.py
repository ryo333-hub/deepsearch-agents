"""Pure, bounded conversion of ranked RetrievalHit records into cited evidence.

The E5 tokenizer is unrelated to a future answer model's token budget. This
layer therefore uses an explicit per-item character cap and never paraphrases.
"""

from collections.abc import Sequence

from app.local_rag.config import MAX_EVIDENCE_TEXT_LENGTH
from app.local_rag.schemas import Evidence, RetrievalHit


class EvidenceBuildError(ValueError):
    """A retrieval hit cannot be cited accurately or violates the budget."""


def build_evidence(hits: Sequence[RetrievalHit], *,
                   max_evidence_chars: int = MAX_EVIDENCE_TEXT_LENGTH) -> tuple[Evidence, ...]:
    """Preserve input order and scores; assign C1..Cn in this result only.

    Truncation is an exact prefix of the Chunk text. Character offsets refer
    to the loader's normalized source block, as in the existing Citation API.
    """
    if (not isinstance(hits, Sequence) or isinstance(hits, (str, bytes))
            or type(max_evidence_chars) is not int
            or not 1 <= max_evidence_chars <= MAX_EVIDENCE_TEXT_LENGTH):
        raise EvidenceBuildError("Invalid hits or max_evidence_chars")

    result = []
    seen = set()
    knowledge_base_id = None
    for position, raw in enumerate(hits, start=1):
        if not isinstance(raw, RetrievalHit):
            raise EvidenceBuildError("Every item must be a RetrievalHit")
        try:
            hit = RetrievalHit.model_validate(raw.model_dump(warnings=False))
        except ValueError:
            raise EvidenceBuildError("RetrievalHit or citation is invalid") from None
        cited = hit.citation
        if hit.chunk_id in seen:
            raise EvidenceBuildError("Duplicate chunk_id in retrieval hits")
        seen.add(hit.chunk_id)
        if knowledge_base_id is None:
            knowledge_base_id = cited.knowledge_base_id
        elif cited.knowledge_base_id != knowledge_base_id:
            raise EvidenceBuildError("Retrieval hits span multiple knowledge bases")
        if (cited.score is None or cited.score_type is None
                or not hit.text or not hit.text.strip()):
            raise EvidenceBuildError("Retrieval hit has no text or score metadata")
        if (cited.start_char is not None
                and cited.end_char - cited.start_char != len(hit.text)):
            raise EvidenceBuildError("Citation does not describe the full retrieval text")
        if cited.start_line is not None:
            # The source Chunker counts physical newlines in its emitted slice.
            actual_end_line = cited.start_line + hit.text[:-1].count("\n")
            if cited.end_line != actual_end_line:
                raise EvidenceBuildError("Citation line range does not match retrieval text")

        text = hit.text[:max_evidence_chars]
        truncated = len(text) < len(hit.text)
        if truncated and cited.start_char is None:
            raise EvidenceBuildError("Cannot truncate without exact source character offsets")
        changes = {"citation_id": f"C{position}"}
        if truncated:
            changes["end_char"] = cited.start_char + len(text)
            if cited.start_line is not None:
                changes["end_line"] = cited.start_line + text[:-1].count("\n")
        try:
            citation = cited.model_copy(update=changes)
            result.append(Evidence(text=text, citation=citation, truncated=truncated))
        except ValueError:
            raise EvidenceBuildError("Evidence citation range cannot be represented") from None
    return tuple(result)


def format_evidence_for_context(evidence: Sequence[Evidence]) -> str:
    """Render already-built evidence deterministically; no ranking or IO."""
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise EvidenceBuildError("evidence must be a sequence")
    parts = []
    seen = set()
    for position, raw in enumerate(evidence, start=1):
        if not isinstance(raw, Evidence):
            raise EvidenceBuildError("Every item must be Evidence")
        try:
            item = Evidence.model_validate(raw.model_dump(warnings=False))
        except ValueError:
            raise EvidenceBuildError("Evidence or citation is invalid") from None
        c = item.citation
        if c.citation_id != f"C{position}" or c.chunk_id in seen:
            raise EvidenceBuildError("Evidence citation IDs must be sequential and unique")
        seen.add(c.chunk_id)
        if c.score is None or c.score_type is None:
            raise EvidenceBuildError("Evidence has no retrieval score")
        location = []
        if c.page is not None:
            location.append(f"第 {c.page} 页")
        if c.start_line is not None:
            location.append(f"第 {c.start_line}-{c.end_line} 行")
        if c.heading is not None:
            location.append(f"标题「{c.heading}」")
        if c.source_block_index is not None:
            location.append(f"块 {c.source_block_index}，字符 [{c.start_char},{c.end_char})")
        lines = [f"[{c.citation_id}]", f"来源：{c.document_name}",
                 f"知识库：{c.knowledge_base_id}，文档：{c.document_id}，片段：{c.chunk_id}"]
        if location:
            lines.append("位置：" + " / ".join(location))
        lines += [f"相关度（{c.score_type}）：{c.score:.6f}",
                  f"已裁剪：{'是' if item.truncated else '否'}", f"内容：\n{item.text}"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
