"""
Streamlit demo UI for MedGuard RAG.

Talks to the FastAPI backend (app/api/main.py) rather than importing the
pipeline directly — this is deliberate: it proves the API actually works
as a real client-server system, not just as a monolithic script, and it
means the UI could be deployed separately from the backend if needed.

Run the backend first:
    uvicorn app.api.main:app --port 8000

Then run this:
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="MedGuard RAG",
    page_icon="💊",
    layout="wide",
)

# --- Header ---
st.title("💊 MedGuard RAG")
st.caption(
    "Drug Interaction & Dosage Clinical Decision Support — "
    "every answer grounded in cited evidence from DDInter + FDA labels."
)

# --- Sidebar: mode selector + info ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool retrieves real drug interaction data (DDInter 2.0) "
        "and FDA label text (openFDA), then generates a cited answer "
        "using Claude. It is **not** a substitute for professional "
        "clinical judgment."
    )
    st.divider()

    mode = st.radio(
        "Query mode",
        ["Full RAG (retrieval + AI answer)", "Search only (no AI cost)", "Direct interaction lookup"],
        help="'Search only' skips the Claude API call — useful for exploring what the retrieval layer finds without spending API credits.",
    )

    st.divider()
    st.markdown("**System status**")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"API connected — {health['qdrant_points']:,} chunks indexed")
    except Exception:
        st.error("API not reachable. Is the FastAPI backend running on port 8000?")

# --- Main input ---
if mode == "Direct interaction lookup":
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.text_input("Drug A", placeholder="e.g. Warfarin")
    with col2:
        drug_b = st.text_input("Drug B", placeholder="e.g. Ibuprofen")
    submit = st.button("Check interaction", type="primary", use_container_width=True)
else:
    question = st.text_area(
        "Ask a clinical question",
        placeholder="e.g. Does Warfarin interact with Ibuprofen? / Is Indomethacin safe for kidneys?",
        height=80,
    )
    submit = st.button("Submit", type="primary", use_container_width=True)

# --- Results ---
if submit:

    # --- Direct interaction lookup ---
    if mode == "Direct interaction lookup":
        if not drug_a or not drug_b:
            st.warning("Please enter both drug names.")
        else:
            with st.spinner("Looking up interaction..."):
                resp = requests.post(
                    f"{API_BASE}/interactions",
                    json={"drug_a": drug_a, "drug_b": drug_b},
                    timeout=10,
                )
            if resp.status_code == 200:
                data = resp.json()
                severity = data["severity"]
                color = {"Major": "🔴", "Moderate": "🟡", "Minor": "🟢"}.get(severity, "⚪")
                st.markdown(f"### {color} {data['drug_a']} + {data['drug_b']}: **{severity}**")
                if data.get("mechanism"):
                    st.markdown(f"**Mechanism:** {data['mechanism']}")
                if data.get("management"):
                    st.markdown(f"**Management:** {data['management']}")
                st.caption(f"Source: {data['source']}")
            elif resp.status_code == 404:
                st.info(resp.json().get("detail", "No interaction record found."))
            else:
                st.error(f"API error: {resp.status_code}")

    # --- Search only ---
    elif mode == "Search only (no AI cost)":
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching..."):
                resp = requests.post(
                    f"{API_BASE}/search",
                    json={"question": question},
                    timeout=15,
                )
            if resp.status_code == 200:
                data = resp.json()

                if data.get("matched_drugs"):
                    st.markdown(f"**Matched drugs:** {', '.join(data['matched_drugs'])}")

                if data.get("structured_interaction"):
                    si = data["structured_interaction"]
                    severity = si["severity"]
                    color = {"Major": "🔴", "Moderate": "🟡", "Minor": "🟢"}.get(severity, "⚪")
                    st.markdown(f"### {color} Structured interaction: {si['drug_a']} + {si['drug_b']} — **{severity}**")

                st.markdown("### Retrieved evidence")
                for i, chunk in enumerate(data["retrieved_chunks"], 1):
                    with st.expander(
                        f"[{chunk['score']:.3f}] {chunk['drug_name']} — {chunk['section']}", expanded=(i <= 2)
                    ):
                        st.markdown(chunk["text"])
            else:
                st.error(f"API error: {resp.status_code}")

    # --- Full RAG ---
    else:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving evidence and generating answer..."):
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question},
                    timeout=60,
                )
            if resp.status_code == 200:
                data = resp.json()

                if data.get("matched_drugs"):
                    st.markdown(f"**Matched drugs:** {', '.join(data['matched_drugs'])}")

                if data.get("structured_interaction"):
                    si = data["structured_interaction"]
                    severity = si["severity"]
                    color = {"Major": "🔴", "Moderate": "🟡", "Minor": "🟢"}.get(severity, "⚪")
                    st.info(f"{color} Structured severity: {si['drug_a']} + {si['drug_b']} — **{severity}**")

                if not data.get("evidence_sufficient"):
                    st.warning("⚠️ Limited evidence available — answer may be incomplete.")

                st.markdown("### Answer")
                st.markdown(data["answer"])

                with st.expander("Retrieved evidence chunks", expanded=False):
                    for i, chunk in enumerate(data["retrieved_chunks"], 1):
                        st.markdown(
                            f"**[{chunk['score']:.3f}] {chunk['drug_name']} — {chunk['section']}**"
                        )
                        st.markdown(chunk["text"])
                        st.divider()

            elif resp.status_code == 503:
                st.error("Claude API key not configured on the backend.")
            else:
                st.error(f"API error: {resp.status_code}")

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This is a portfolio/educational project. Not FDA-approved, not a "
    "medical device, not a substitute for professional clinical judgment. "
    "Always verify against current, authoritative sources."
)
