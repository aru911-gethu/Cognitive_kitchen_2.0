import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

from cognitive_kitchen.retrieval.chain import (
    get_vector_store,
    format_docs,
    CHEF_PROMPT_TEMPLATE,
    get_hf_llm,
    get_openai_llm,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

API_BASE_URL = "http://127.0.0.1:8000"
CHEF_AVATAR_URL = "https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSRriFUG0P4RqHcoVF461b4GAaThLEhcy7bFZW1LeIB-KKNKYQ1RG_46DNDf-OixuPMMR4855mjZF4iE_Q"

st.set_page_config(
    page_title="CognitiveKitchen",
    page_icon="👨‍🍳",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .chef-hero {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 1rem 1.4rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .chef-img {
        width: 65px;
        height: 65px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #10b981;
    }
    .chunk-box {
        background-color: #0f172a;
        border-left: 4px solid #ef4444;
        padding: 0.8rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding-bottom: 1rem;">
            <img src="{CHEF_AVATAR_URL}" style="width: 90px; height: 90px; border-radius: 50%; border: 3px solid #10b981;">
            <h3 style="margin: 0.5rem 0 0 0;">Chef Pierre</h3>
            <p style="color: #94a3b8; font-size: 0.85rem;">Your AI Culinary Companion</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    menu_choice = st.radio(
        "Workspace",
        ["💬 Kitchen Chat (Side-by-Side)", "📥 Feed the Chef (Ingest)", "🛠️ Diagnostics & API"],
    )

    if menu_choice == "💬 Kitchen Chat (Side-by-Side)":
        st.markdown("---")
        st.markdown("### 🧪 RAG Engine Config")
        index_name = st.text_input("FAISS Vector Index", value="test_naive_experiment")
        k_val = st.slider("Retrieved Chunks (k)", min_value=1, max_value=4, value=2)

# ========================================================
# TAB 1: KITCHEN CHAT (SIDE-BY-SIDE AUDIT LAB)
# ========================================================
if menu_choice == "💬 Kitchen Chat (Side-by-Side)":
    st.markdown(
        f"""
        <div class="chef-hero">
            <img src="{CHEF_AVATAR_URL}" class="chef-img">
            <div>
                <h2 style="margin: 0; font-size: 1.4rem;">Cognitive Kitchen: Side-by-Side RAG Audit</h2>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
                    Compare how <b>Qwen2.5-0.5B</b> exposes context bleed vs. how <b>gpt-4o-mini</b> attempts to mask it.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    if c1.button("🧪 Trigger Bleed: Murukku Open Sandwich"):
        st.session_state.quick_query = "How do I prepare and assemble the Chennai Special Murukku Open Sandwich?"
    if c2.button("🥒 Standard Query: Fresh Cucumber Raita"):
        st.session_state.quick_query = "How do I make Fresh Cucumber Raita?"

    prompt = st.chat_input("Ask Chef Pierre about a recipe...") or st.session_state.pop("quick_query", None)

    if prompt:
        st.subheader(f"Query: *\"{prompt}\"*")

        try:
            vector_store = get_vector_store(index_name)
            retriever = vector_store.as_retriever(search_kwargs={"k": k_val})
            retrieved_docs = retriever.invoke(prompt)
            context_str = format_docs(retrieved_docs)
        except Exception as e:
            st.error(f"Error accessing FAISS index '{index_name}': {e}. Run `scripts/test_naive_hallucination.py` first.")
            st.stop()

        with st.expander("🔍 Inspect Raw Retrieved Chunks (Shared Context Fed to Both Models)", expanded=True):
            for doc in retrieved_docs:
                cid = doc.metadata.get("chunk_id", "unknown")
                st.markdown(f"**[{cid}]**")
                st.markdown(f"<div class='chunk-box'>{doc.page_content}</div>", unsafe_allow_html=True)

        prompt_tpl = ChatPromptTemplate.from_template(CHEF_PROMPT_TEMPLATE)
        chain_qwen = prompt_tpl | get_hf_llm() | StrOutputParser()
        chain_openai = prompt_tpl | get_openai_llm() | StrOutputParser()

        col_local, col_frontier = st.columns(2)

        with col_local:
            st.error("🤖 Local Model (Qwen2.5-0.5B-Instruct)")
            st.caption("Exposes chunk bleed: relies strictly on ingested text.")
            with st.spinner("Qwen is generating..."):
                st.write_stream(chain_qwen.stream({"context": context_str, "question": prompt}))

        with col_frontier:
            st.success("🧠 Frontier API (OpenAI gpt-4o-mini)")
            st.caption("Masks chunk bleed: parametric pre-training overrides bad context.")
            with st.spinner("OpenAI is generating..."):
                st.write_stream(chain_openai.stream({"context": context_str, "question": prompt}))

# ========================================================
# TAB 2: FEED THE CHEF (INGEST)
# ========================================================
elif menu_choice == "📥 Feed the Chef (Ingest)":
    st.markdown("### 📥 Add New Recipes")
    st.caption("Feed recipes via PDF or URL without dealing with raw markup or technical selectors.")

    source_type = st.radio("Choose Input Format", ["PDF Document", "Web Page / URL"], horizontal=True)

    if source_type == "PDF Document":
        uploaded_pdf = st.file_uploader("Upload Cookbook PDF", type=["pdf"])
        if uploaded_pdf and st.button("Process Document", use_container_width=True):
            with st.spinner("Extracting recipe layout..."):
                try:
                    files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_BASE_URL}/ingest/pdf", files=files, timeout=45)
                    if resp.status_code == 200:
                        data = resp.json()
                        total_blocks = data.get("total_blocks", 0)
                        saved_to = data.get("saved_to", "")
                        blocks = data.get("blocks", [])
                        page_count = max([b["page"] for b in blocks]) if blocks else 0

                        st.success(f"Processed '{uploaded_pdf.name}' successfully!")
                        col1, col2 = st.columns(2)
                        col1.metric("Total Pages", page_count)
                        col2.metric("Parsed Structural Blocks", total_blocks)
                        if saved_to:
                            st.caption(f"Stored dataset: `{saved_to}`")
                    else:
                        err_detail = resp.json().get("detail", resp.text)
                        st.error(f"Failed to parse PDF: {err_detail}")
                except Exception as e:
                    st.error(f"Backend connection error: {e}")

    else:
        url_input = st.text_input(
            "Recipe or Collection Web Address",
            value="https://cookpad.com/in/search/chennai",
            placeholder="Paste any recipe link or catalog page here...",
        )
        max_items = st.slider("Target Recipe Count", min_value=1, max_value=15, value=5)

        if st.button("Extract Recipes Automatically", use_container_width=True):
            if not url_input.strip() or not url_input.startswith("http"):
                st.warning("Please enter a valid website address starting with http:// or https://")
            else:
                progress_box = st.empty()
                progress_box.info("Chef Pierre is scanning the page and finding recipe links...")
                try:
                    payload = {
                        "category_url": url_input.strip(),
                        "link_selector": "",
                        "max_items": max_items,
                    }
                    resp = requests.post(f"{API_BASE_URL}/ingest/url", json=payload, timeout=90)
                    if resp.status_code == 200:
                        data = resp.json()
                        headers = data.get("recipe_headers", [])
                        count = data.get("extracted_count", 0)

                        progress_box.empty()
                        st.success(f"Extracted {count} recipes successfully!")
                        if data.get("saved_to"):
                            st.caption(f"Stored dataset: `{data.get('saved_to')}`")

                        st.markdown("#### Ingested Recipe Collection")
                        if headers:
                            for idx, h in enumerate(headers, start=1):
                                st.markdown(f"**{idx}.** {h}")
                        else:
                            st.info("No recipes found on this page. Try another link.")
                    else:
                        progress_box.empty()
                        st.error(f"Extraction failed: {resp.text}")
                except Exception as e:
                    progress_box.empty()
                    st.error(f"Backend connection error: {e}")

# ========================================================
# TAB 3: DIAGNOSTICS & API
# ========================================================
elif menu_choice == "🛠️ Diagnostics & API":
    st.markdown("### 🛠️ Diagnostics & API Suite")

    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        api_up = health_resp.status_code == 200
    except Exception:
        api_up = False

    c1, c2 = st.columns(2)
    c1.metric("FastAPI Backend", "Online (Port 8000)" if api_up else "Offline")
    c2.markdown(f"**Interactive Docs:** [Swagger OpenAPI Specification]({API_BASE_URL}/docs)")

    st.markdown("---")
    st.markdown("#### Chunking Failure Analysis")
    sample_text = (
        "RECIPE: Quick Jeera Rice\n"
        "Ingredients: 1 cup basmati rice, 1 tsp cumin seeds, 2 cups water, 1 tbsp ghee.\n"
        "Instructions: Wash the rice. Heat ghee in a pressure cooker. Add cumin seeds.\n"
        "Add drained rice and water. Close the lid tightly and pressure cook for 4 whistles.\n\n"
        "RECIPE: Fresh Cucumber Raita\n"
        "Ingredients: 1 cup chilled yogurt, 1 grated cucumber, 1/4 tsp roasted cumin powder, salt to taste.\n"
        "Instructions: Whisk the yogurt in a bowl until smooth. Fold in grated cucumber and salt.\n"
        "Garnish with roasted cumin powder. Serve immediately chilled."
    )
    raw_doc = st.text_area("Test Document", value=sample_text, height=130)
    col_w, col_o = st.columns(2)
    size = col_w.slider("Window Size", 100, 500, 260, 20)
    overlap = col_o.slider("Overlap", 0, 80, 30, 5)

    if st.button("Run Diagnostic Check", use_container_width=True):
        try:
            res = requests.post(
                f"{API_BASE_URL}/chunk/naive",
                json={"text": raw_doc, "chunk_size": size, "chunk_overlap": overlap},
            )
            if res.status_code == 200:
                chunks = res.json()["chunks"]
                contaminated = sum(
                    1 for c in chunks if "pressure cook" in c["text"] and "Cucumber Raita" in c["text"]
                )

                m1, m2 = st.columns(2)
                m1.metric("Total Chunks", len(chunks))
                m2.metric(
                    "Cross-Contaminated",
                    f"{contaminated} ({contaminated/len(chunks)*100:.0f}%)",
                )

                for c in chunks:
                    is_bad = "pressure cook" in c["text"] and "Cucumber Raita" in c["text"]
                    with st.expander(f"{'🔴 Contaminated' if is_bad else '🟢 Clean'} [{c['chunk_id']}]"):
                        st.code(c["text"])
        except Exception as e:
            st.error(f"Error calling diagnostics API: {e}")