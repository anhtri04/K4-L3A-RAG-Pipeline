import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Football RAG Chatbot",
    page_icon="⚽",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected" not in st.session_state:
    st.session_state.selected = None
if "msg_seq" not in st.session_state:
    st.session_state.msg_seq = 0

with st.sidebar:
    st.title("⚽ Football RAG")
    st.caption("Luật IFAB + tin bóng đá. Trả lời kèm citation.")
    top_k = st.slider("Số chunks", 3, 10, 5)
    st.divider()
    if st.button("Xóa hội thoại"):
        st.session_state.messages = []
        st.session_state.selected = None
        st.rerun()

st.title("⚽ Football RAG Chatbot")
st.caption("Hỏi về luật (việt vị, handball, VAR, thẻ phạt) hoặc chiến thuật. Click citation [1][2] để xem tài liệu ở panel phải.")

chat_col, doc_col = st.columns([2, 1])

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption(f"retrieval: {msg.get('retrieval_source','-')} | {len(msg['sources'])} sources")
                cols = st.columns(min(len(msg["sources"]), 5))
                for i, src in enumerate(msg["sources"]):
                    label = f"[{i+1}] {src['metadata'].get('title','')[:24]}"
                    if cols[i % len(cols)].button(label, key=f"cite-{msg['id']}-{i}"):
                        st.session_state.selected = src
                        st.rerun()

    query = st.chat_input("Nhập câu hỏi về bóng đá...")

    if query:
        from src.task10_generation import generate_with_citation

        st.session_state.msg_seq += 1
        st.session_state.messages.append({"role": "user", "content": query, "id": st.session_state.msg_seq})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang检索 + sinh câu trả lời..."):
                try:
                    result = generate_with_citation(query, top_k=top_k)
                except Exception as error:
                    result = {
                        "answer": f"Lỗi pipeline: {error}",
                        "sources": [],
                        "retrieval_source": "none",
                    }
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption(f"retrieval: {result['retrieval_source']} | {len(result['sources'])} sources")
                cols = st.columns(min(len(result["sources"]), 5))
                for i, src in enumerate(result["sources"]):
                    label = f"[{i+1}] {src['metadata'].get('title','')[:24]}"
                    if cols[i % len(cols)].button(label, key=f"cite-new-{st.session_state.msg_seq}-{i}"):
                        st.session_state.selected = src
                        st.rerun()

        st.session_state.msg_seq += 1
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "id": st.session_state.msg_seq,
        })
        st.rerun()

with doc_col:
    st.subheader("📄 Source panel")
    src = st.session_state.selected
    if src is None:
        st.caption("Click một citation [1][2]... để xem tài liệu gốc tại đây.")
    else:
        meta = src.get("metadata", {})
        st.markdown(f"**{meta.get('title','')}**")
        st.caption(
            f"{meta.get('source','')} | chunk {meta.get('chunk_index','-')} | "
            f"{src.get('retrieval_method','')} | score {src.get('score',0):.3f}"
        )
        st.markdown(src.get("content", ""))
        if st.button("Đóng panel"):
            st.session_state.selected = None
            st.rerun()
