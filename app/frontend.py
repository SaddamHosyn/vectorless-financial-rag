import sys
from pathlib import Path

# Add project root directory to sys.path for Streamlit Cloud deployment
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from app.main import generate_answer, retrieve_chunks, embed_query
from app.entity_resolver import resolve_form


st.set_page_config(page_title="Financial & Policy RAG Assistant", page_icon="🏦")
st.title("🏦 Financial & Policy RAG Assistant")
st.caption("Ask questions about loan terms, policies, repayment procedures, and account services")


if "history" not in st.session_state:
    st.session_state.history = []


show_chunks = st.sidebar.checkbox("Show retrieved chunks (debug mode)", value=False)
if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.rerun()


with st.form(key="ask_form"):
    question = st.text_input(
        "Ask a question:", placeholder="e.g. What is the procedure for early loan repayment?"
    )
    submitted = st.form_submit_button("Ask")


if submitted and question.strip():
    with st.spinner("Searching knowledge base and generating response..."):
        query_embedding = embed_query(question)
        chunks = retrieve_chunks(query_embedding)
        answer = generate_answer(question, chunks=chunks)
        form_match = resolve_form(question)

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer.get("answer", ""),
            "chunks": chunks,
            "form_match": form_match,
            "latency_ms": answer.get("latency_ms", 0),
            "cached": answer.get("cached", False),
            "estimated_cost_usd": answer.get("estimated_cost_usd", 0),
        }
    )


for entry in reversed(st.session_state.history):
    st.markdown(f"**Q:** {entry['question']}")
    st.markdown(f"**A:** {entry['answer']}")
    st.caption(f"⏱ {entry['latency_ms']:.0f}ms | 💰 ${entry['estimated_cost_usd']:.6f} | {'⚡ Cached' if entry['cached'] else '🔍 Live'}")

    if entry["form_match"]:
        st.info(f"Related document/form: {entry['form_match']['form_name']}")

    if show_chunks:
        with st.expander("Retrieved chunks"):
            if not entry["chunks"]:
                st.write("No chunks retrieved.")
            for text, filename, similarity in entry["chunks"]:
                st.markdown(f"**{filename}** (similarity: {similarity:.3f})")
                st.text(text[:500] + ("..." if len(text) > 500 else ""))

    st.divider()
