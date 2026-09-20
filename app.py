"""Streamlit interface for the football RAG pipeline."""

from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def _score_label(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _render_source_buttons(
    sources: list[dict],
    *,
    message_id: int,
    retrieval_source: str,
) -> None:
    """Show citation indexes plus the retrieval method and score."""
    st.caption(f"Retrieval: {retrieval_source} · {len(sources)} nguồn")
    columns = st.columns(min(len(sources), 3))
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = str(metadata.get("title") or metadata.get("source") or "Untitled")
        method = source.get("retrieval_method", "-")
        score = _score_label(source.get("score"))
        with columns[(index - 1) % len(columns)]:
            st.caption(f"[{index}] {method} · {score}")
            if st.button(
                title[:40],
                key=f"source-{message_id}-{index}",
                use_container_width=True,
            ):
                st.session_state.selected_source = {
                    "citation_index": index,
                    "source": source,
                }
                st.rerun()


def _render_source_panel(selection: dict | None) -> None:
    st.subheader("📄 Chi tiết nguồn")
    if not selection:
        st.caption("Chọn một nguồn [1], [2]… dưới câu trả lời để xem bằng chứng.")
        return

    source = selection.get("source", selection)
    citation_index = selection.get("citation_index", "-")
    metadata = source.get("metadata", {})
    title = metadata.get("title") or "Untitled"
    st.markdown(f"### [{citation_index}] {title}")
    st.caption(
        " · ".join(
            (
                f"Tệp: {metadata.get('source', '-')}",
                f"Loại: {metadata.get('doc_type', '-')}",
                f"Chunk: {metadata.get('chunk_index', '-')}",
                f"Phương thức: {source.get('retrieval_method', '-')}",
                f"Điểm: {_score_label(source.get('score'))}",
            )
        )
    )
    url = metadata.get("url")
    if isinstance(url, str) and urlparse(url).scheme in {"http", "https"}:
        st.link_button("Mở tài liệu gốc", url)
    st.divider()
    st.markdown(str(source.get("content") or "_Không có nội dung để hiển thị._"))
    if st.button("Đóng chi tiết nguồn", use_container_width=True):
        st.session_state.selected_source = None
        st.rerun()


st.set_page_config(
    page_title="Football RAG Chatbot",
    page_icon="⚽",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None
if "message_sequence" not in st.session_state:
    st.session_state.message_sequence = 0

with st.sidebar:
    st.title("⚽ Football RAG")
    st.caption(
        "Hỏi đáp trên bộ tài liệu về quản trị, an toàn, bóng đá nữ và "
        "nghiên cứu phân tích bóng đá."
    )
    top_k = st.slider("Số đoạn bằng chứng", 3, 10, 5)
    st.info("Mỗi citation [n] ánh xạ trực tiếp tới nguồn [n] ở câu trả lời.")
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_source = None
        st.rerun()

st.title("⚽ Football RAG Chatbot")
st.caption(
    "Ví dụ: Football Governance Act bảo vệ người hâm mộ như thế nào? "
    "Hướng dẫn chấn động trong thể thao đề xuất điều gì?"
)

chat_column, source_column = st.columns([2, 1], gap="large")

with chat_column:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                _render_source_buttons(
                    message["sources"],
                    message_id=message["id"],
                    retrieval_source=message.get("retrieval_source", "-"),
                )

    query = st.chat_input("Nhập câu hỏi về bộ tài liệu bóng đá…")
    if query:
        from src.task10_generation import generate_with_citation

        st.session_state.message_sequence += 1
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "id": st.session_state.message_sequence,
            }
        )
        with st.spinner("Đang truy xuất bằng chứng và tạo câu trả lời…"):
            try:
                result = generate_with_citation(query, top_k=top_k)
            except Exception:
                # Never expose provider internals or credentials in the UI.
                result = {
                    "answer": SAFE_REFUSAL,
                    "sources": [],
                    "retrieval_source": "none",
                }

        st.session_state.message_sequence += 1
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer") or SAFE_REFUSAL,
                "sources": result.get("sources") or [],
                "retrieval_source": result.get("retrieval_source", "none"),
                "id": st.session_state.message_sequence,
            }
        )
        st.rerun()

with source_column:
    _render_source_panel(st.session_state.selected_source)
