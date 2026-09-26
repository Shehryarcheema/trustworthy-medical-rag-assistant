# =====================================================================
#  TRUSTWORTHY MEDICAL RAG ASSISTANT — Professional GUI (v6)
#  MSc Artificial Intelligence, De Montfort University
#
#  Backend logic mirrors the Colab notebook (protected core).
#  Token-efficient Groq setup (reasoning_effort=low for gpt-oss),
#  model auto-pick, retry, persistent answer, history, download.
# =====================================================================
import os
import html
import time
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Trustworthy Medical RAG Assistant",
                   page_icon="🩺", layout="centered")

st.markdown("""
<style>
  .main .block-container {max-width: 860px; padding-top: 1.2rem;}
  .hero {background: linear-gradient(135deg,#1F3864 0%,#2E74B5 100%);
         border-radius: 14px; padding: 22px 26px; margin-bottom: 6px;}
  .hero h1 {color:#fff; font-size:1.6rem; font-weight:800; margin:0 0 4px 0;}
  .hero p  {color:#D6E4F5; margin:0; font-size:.95rem;}
  .sec-head {font-size:1.05rem; font-weight:800; color:#1F3864;
             border-bottom:3px solid #2E74B5; padding-bottom:4px;
             margin:26px 0 12px 0; letter-spacing:.4px; text-transform:uppercase;}
  .q-box {background:#EEF3FA; border:1px solid #C9D8EC; border-radius:10px;
          padding:10px 14px; color:#1F3864; font-weight:600;}
  .answer-panel {background:#FFFFFF; border:1px solid #C9D8EC; border-left:6px solid #2E74B5;
                 border-radius:10px; padding:18px 20px; font-size:1.05rem; line-height:1.65;
                 color:#1a1a1a; max-height:430px; overflow-y:auto;
                 box-shadow:0 2px 8px rgba(31,56,100,.07);}
  .conf-label {font-size:1.0rem; font-weight:800; letter-spacing:1px;}
  .meter-track {background:#E9EDF2; border-radius:9px; height:20px; width:100%; overflow:hidden;}
  .meter-fill  {height:20px; border-radius:9px;}
  .mini-track {background:#E9EDF2; border-radius:6px; height:11px; width:100%; overflow:hidden;}
  .mini-fill  {height:11px; border-radius:6px;}
  .comp-label {font-size:.82rem; color:#374151; margin-bottom:2px; font-weight:600;}
  .ret-row {display:flex; align-items:center; gap:10px; margin-bottom:7px;}
  .ret-name {min-width:110px; font-size:.85rem; color:#1F3864; font-weight:700;}
  .ret-val {min-width:46px; font-size:.85rem; color:#374151; font-weight:600; text-align:right;}
  .disc {color:#7a5d00; background:#FFF7E0; border:1px solid #F0D77A;
         padding:10px 14px; border-radius:10px; font-size:.86rem; margin-top:18px;}
  .src-q {color:#1F3864; font-weight:700;}
  div[data-testid="stMetricValue"] {font-size:1.25rem; color:#1F3864;}
</style>
""", unsafe_allow_html=True)

ASSET_DIR   = "rag_assets"
INDEX_DIR   = os.path.join(ASSET_DIR, "faiss_index_medquad")
PARENTS_PKL = os.path.join(ASSET_DIR, "parents.pkl")
EMB_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
REFUSAL     = "I cannot answer this from the provided documents."
ANSWER_CAP  = 1000

if "result" not in st.session_state:  st.session_state.result = None
if "history" not in st.session_state: st.session_state.history = []

def clear_result():
    st.session_state.result = None
    st.session_state.q_input = ""

def use_example(text):
    st.session_state.q_input = text
    st.session_state.auto_ask = True

with st.sidebar:
    st.header("⚙️ Settings")
    try:
        key_from_secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key_from_secret = ""
    api_key = key_from_secret or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Free key at console.groq.com")
    n_samples = st.slider("Confidence samples (N)", 2, 5, 2)
    conf_threshold = st.slider("Low-confidence warning below (%)", 30, 80, 60)
    k_parents = st.slider("Sources retrieved", 2, 6, 3)

    st.markdown("---")
    st.subheader("🕘 Question history")
    if st.session_state.history:
        for j, h in enumerate(reversed(st.session_state.history[-10:])):
            icon = "✅" if h["type"] == "ok" else "🚫"
            label = f"{icon} {h['q'][:34]}" + ("…" if len(h["q"]) > 34 else "")
            if h["type"] == "ok" and h.get("conf") is not None:
                label += f"  ({h['conf']:.0f}%)"
            st.button(label, key=f"hist_{j}", use_container_width=True,
                      on_click=use_example, args=(h["q"],))
        st.caption("Click a past question to ask it again.")
    else:
        st.caption("No questions asked yet.")
    st.markdown("---")
    st.caption("MSc dissertation: *Trustworthy RAG Assistant with Confidence Scoring* — DMU.")

@st.cache_resource(show_spinner="Loading the medical knowledge base…")
def load_assets():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from sentence_transformers import SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL,
                                       encode_kwargs={"normalize_embeddings": True})
    index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    st_model = SentenceTransformer(EMB_MODEL)
    with open(PARENTS_PKL, "rb") as f:
        parents = pickle.load(f)
    return index, st_model, parents

@st.cache_resource(show_spinner="Connecting to Groq…")
def get_llms(api_key):
    os.environ["GROQ_API_KEY"] = api_key
    import groq
    from langchain_groq import ChatGroq
    ids = sorted(m.id for m in groq.Groq().models.list().data)
    prefer = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
              "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    model = next((m for m in prefer if m in ids),
                 next((m for m in ids if "llama" in m.lower() and "guard" not in m.lower()),
                      "openai/gpt-oss-20b"))
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    return (ChatGroq(model=model, temperature=0.2, max_tokens=450, **extra),
            ChatGroq(model=model, temperature=0.8, max_tokens=350, **extra), model)

from sentence_transformers import util
import groq as _groq

def is_refusal(text):
    if not text: return False
    t = text.lower()
    return ("cannot answer" in t) or ("can't answer" in t) or \
           (("not contain" in t or "no information" in t) and len(t) < 220)

def _invoke(llm, prompt, tries=3):
    for k in range(tries):
        try:
            return llm.invoke(prompt).content.strip()
        except _groq.RateLimitError:
            if k == tries - 1:
                return None
            time.sleep(5 * (k + 1))

def retrieve(question, index, st_model, parents, k_parents=3, k_search=14):
    query = "Question: " + question
    docs = index.similarity_search(query, k=k_search)
    q_vec  = st_model.encode(query, normalize_embeddings=True)
    d_vecs = st_model.encode([d.page_content for d in docs], normalize_embeddings=True)
    sims = util.cos_sim(q_vec, d_vecs)[0].tolist()
    best = {}
    for d, s in zip(docs, sims):
        pid = d.metadata["pid"]
        if s > best.get(pid, -1): best[pid] = s
    ranked = sorted(best.items(), key=lambda x: -x[1])[:k_parents]
    picked = [{**parents[pid], "sim": round(s, 3)} for pid, s in ranked]
    return picked, (picked[0]["sim"] if picked else 0.0)

def build_prompt(question, picked):
    blocks = [f"[Source {i}] (from: {p['source']})\nQ: {p['question']}\nA: {p['answer'][:ANSWER_CAP]}"
              for i, p in enumerate(picked, 1)]
    context = "\n\n".join(blocks)
    return ("You are a careful medical information assistant.\n"
            "Using ONLY the sources below, give a clear, complete answer to the user's question. "
            "Combine information from several sources if that helps. "
            "If the sources cover the topic only partly, answer with what is available and say what is missing. "
            f"Only if none of the sources are relevant, reply exactly: '{REFUSAL}'\n"
            "Never add medical facts that are not in the sources.\n\n"
            f"{context}\n\nUser question: {question}\nAnswer:")

def generate(question, picked, llm):
    return _invoke(llm, build_prompt(question, picked))

def confidence_components(question, picked, primary, st_model, llm_sample, n=2):
    samples = []
    for _ in range(n):
        s = generate(question, picked, llm_sample)
        if s: samples.append(s)
        time.sleep(0.3)
    real = [a for a in ([primary] + samples) if a and not is_refusal(a)]
    top_sim = picked[0]["sim"]
    evidence = float(np.clip((top_sim - 0.35) / (0.80 - 0.35), 0, 1))
    context = " ".join(p["answer"][:ANSWER_CAP] for p in picked)[:3000]
    g = util.cos_sim(st_model.encode(primary, normalize_embeddings=True),
                     st_model.encode(context, normalize_embeddings=True))[0][0]
    groundedness = float(np.clip((float(g) - 0.2) / (0.8 - 0.2), 0, 1))
    if len(real) >= 2:
        vecs = st_model.encode(real, normalize_embeddings=True)
        S = util.cos_sim(vecs, vecs).numpy(); m = len(real)
        agreement = float((S.sum() - m) / (m * (m - 1)))
        final = round(100 * (0.60*agreement + 0.25*evidence + 0.15*groundedness), 1)
        return {"confidence": final, "agreement": round(agreement*100,1),
                "evidence": round(evidence*100,1), "groundedness": round(groundedness*100,1)}
    else:
        final = round(100 * (0.625*evidence + 0.375*groundedness), 1)
        return {"confidence": final, "agreement": None,
                "evidence": round(evidence*100,1), "groundedness": round(groundedness*100,1)}

def conf_color(v): return "#C00000" if v < 45 else ("#BF9000" if v < 70 else "#2E9E44")
def conf_word(v):  return "LOW CONFIDENCE" if v < 45 else ("MEDIUM CONFIDENCE" if v < 70 else "HIGH CONFIDENCE")
def sec(t): st.markdown(f'<div class="sec-head">{t}</div>', unsafe_allow_html=True)

def meter(value):
    col = conf_color(value)
    st.markdown(f"""
      <div style="display:flex;align-items:center;gap:16px;margin:4px 0;">
        <div style="font-size:2.5rem;font-weight:800;color:{col};min-width:110px;">{value}%</div>
        <div style="flex:1;">
          <div class="meter-track"><div class="meter-fill"
               style="width:{max(2,min(100,value))}%;background:{col};"></div></div>
          <div class="conf-label" style="color:{col};margin-top:6px;">{conf_word(value)}</div>
        </div></div>""", unsafe_allow_html=True)

def mini_bar(label, value):
    col = conf_color(value)
    st.markdown(f"""
      <div class="comp-label">{label} — <b>{value}%</b></div>
      <div class="mini-track"><div class="mini-fill"
           style="width:{max(2,min(100,value))}%;background:{col};"></div></div>
      <div style="height:8px;"></div>""", unsafe_allow_html=True)

def retrieval_bars(picked):
    for i, p in enumerate(picked, 1):
        pct = p["sim"] * 100
        st.markdown(f"""
          <div class="ret-row"><div class="ret-name">Source {i}</div>
            <div style="flex:1;"><div class="mini-track"><div class="mini-fill"
                 style="width:{max(2,min(100,pct))}%;background:#2E74B5;"></div></div></div>
            <div class="ret-val">{p['sim']:.2f}</div></div>""", unsafe_allow_html=True)

def esc(text): return html.escape(text).replace("\n", "<br>")

def make_report(r):
    ag = r['agreement'] if r['agreement'] is not None else "n/a (rate-limited)"
    lines = ["="*60, "TRUSTWORTHY MEDICAL RAG ASSISTANT — ANSWER REPORT", "="*60,
             f"\nQUESTION:\n{r['question']}", f"\nGENERATED ANSWER:\n{r['answer']}",
             f"\nCONFIDENCE: {r['confidence']}%  ({conf_word(r['confidence'])})",
             f"  - Answer agreement:  {ag}", f"  - Evidence strength: {r['evidence']}%",
             f"  - Groundedness:      {r['groundedness']}%",
             f"\nRESPONSE TIME: {r['elapsed']} s   MODEL: {r['model']}", "\nSOURCES:"]
    for i, p in enumerate(r["picked"], 1):
        lines += [f"  [{i}] relevance {p['sim']:.2f} — {p['source']}", f"      Q: {p['question']}"]
    lines.append("\nDisclaimer: AI research prototype — not professional medical advice.")
    return "\n".join(lines)

st.markdown("""
<div class="hero">
  <h1>🩺 TRUSTWORTHY MEDICAL RAG ASSISTANT</h1>
  <p>Grounded answers from the trusted MedQuAD (NIH) dataset — with an explainable
     confidence score, low-confidence warnings, honest refusals, and cited evidence.</p>
</div>""", unsafe_allow_html=True)

if not os.path.isdir(INDEX_DIR) or not os.path.exists(PARENTS_PKL):
    st.error("**Knowledge base not found.** Put the `rag_assets` folder "
             "(with `faiss_index_medquad/` and `parents.pkl`) next to this app.")
    st.stop()

index, st_model, parents = load_assets()
st.caption(f"✅ Knowledge base loaded — {len(parents):,} medical Q&A pairs indexed.")

sec("Ask your medical question")
question = st.text_input("question", key="q_input", label_visibility="collapsed",
                         placeholder="e.g. What are the symptoms of high blood pressure?")
c1, c2 = st.columns([3, 1])
ask_clicked = c1.button("🩺  ASK QUESTION", type="primary", use_container_width=True)
c2.button("🧹  Clear", use_container_width=True, on_click=clear_result)
ex1, ex2, ex3 = st.columns(3)
ex1.button("Treatments for glaucoma", use_container_width=True,
           on_click=use_example, args=("What are the treatments for glaucoma?",))
ex2.button("Causes of type 2 diabetes", use_container_width=True,
           on_click=use_example, args=("What causes type 2 diabetes?",))
ex3.button("Symptoms of asthma", use_container_width=True,
           on_click=use_example, args=("What are the symptoms of asthma?",))

if ask_clicked or st.session_state.pop("auto_ask", False):
    q = st.session_state.q_input.strip()
    if not q:
        st.warning("Please type a question first.")
    elif not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    else:
        try:
            llm_primary, llm_sample, model_name = get_llms(api_key)
            t0 = time.time()
            picked, top_sim = retrieve(q, index, st_model, parents, k_parents=k_parents)
            if top_sim < 0.35:
                st.session_state.result = {"type": "offtopic", "question": q,
                                           "picked": picked, "top_sim": top_sim}
            else:
                with st.spinner("Retrieving evidence and generating a grounded answer…"):
                    primary = generate(q, picked, llm_primary)
                if primary is None:
                    st.session_state.result = {"type": "busy", "question": q}
                elif is_refusal(primary):
                    st.session_state.result = {"type": "refused", "question": q,
                                               "picked": picked, "top_sim": top_sim}
                else:
                    with st.spinner(f"Scoring confidence ({n_samples} samples)…"):
                        comp = confidence_components(q, picked, primary,
                                                     st_model, llm_sample, n=n_samples)
                    st.session_state.result = {
                        "type": "ok", "question": q, "answer": primary, "picked": picked,
                        "top_sim": top_sim, "model": model_name,
                        "elapsed": round(time.time() - t0, 1), **comp}
        except Exception as e:
            st.session_state.result = {"type": "error", "question": q, "msg": str(e)}

        rt = st.session_state.result
        if not st.session_state.history or st.session_state.history[-1]["q"] != q:
            st.session_state.history.append(
                {"q": q, "conf": rt.get("confidence"), "type": rt["type"]})

r = st.session_state.result
if r:
    sec("Your question")
    st.markdown(f'<div class="q-box">❓ {esc(r["question"])}</div>', unsafe_allow_html=True)

    if r["type"] == "error":
        st.error(f"**Something went wrong.**\n\n`{r['msg']}`\n\nCheck your API key and try again.")
    elif r["type"] == "busy":
        st.warning("⏳ **The Groq free daily token limit has been reached.** "
                   "Wait a few minutes, or use a fresh free Groq API key, then ask again. "
                   "(Your pipeline is working — this is only a usage quota.)")
    elif r["type"] == "offtopic":
        sec("Generated answer")
        st.error("🚫 **No relevant information found in the MedQuAD dataset** "
                 f"(best match {r['top_sim']*100:.0f}%). This assistant answers only "
                 "medical questions covered by its knowledge base — it does not guess.")
        with st.expander("Closest topics that ARE in the dataset"):
            for p in r["picked"]:
                st.write(f"• ({p['sim']*100:.0f}%) {p['question']}")
    elif r["type"] == "refused":
        sec("Generated answer")
        st.error("🚫 **I cannot answer this from the provided documents.** "
                 "The retrieved sources do not contain this specific answer.")
        st.write("Closest topic in the dataset: *" + r["picked"][0]["question"] + "*")
    else:
        sec("Generated answer")
        st.markdown(f'<div class="answer-panel">{esc(r["answer"])}</div>', unsafe_allow_html=True)
        if r["confidence"] < conf_threshold:
            st.error(f"⚠️ **Low confidence ({r['confidence']}%)** — this answer may be "
                     "unreliable. Verify with the evidence below or a healthcare professional.")
        sec("Confidence score")
        meter(r["confidence"])
        b1, b2, b3 = st.columns(3)
        ag = r["agreement"] if r["agreement"] is not None else 0
        with b1: mini_bar("Answer agreement" + ("" if r["agreement"] is not None else " (n/a)"), ag)
        with b2: mini_bar("Evidence strength", r["evidence"])
        with b3: mini_bar("Groundedness", r["groundedness"])
        st.caption("confidence = 0.60 × agreement + 0.25 × evidence + 0.15 × groundedness "
                   "(if sampling is rate-limited, it falls back to evidence + groundedness).")
        sec("Supporting evidence")
        for i, p in enumerate(r["picked"], 1):
            with st.expander(f"Source {i}  ·  relevance {p['sim']:.2f}  ·  {p['source']}"):
                st.markdown(f"<span class='src-q'>Q:</span> {esc(p['question'])}", unsafe_allow_html=True)
                st.write(p["answer"][:1200] + ("…" if len(p["answer"]) > 1200 else ""))
        sec("Retrieval analysis")
        retrieval_bars(r["picked"])
        st.caption("Cosine similarity between your question and each retrieved source (real FAISS values).")
        sec("Response metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{r['confidence']}%")
        m2.metric("Response time", f"{r['elapsed']} s")
        m3.metric("Sources used", len(r["picked"]))
        m4.metric("Best match", f"{r['top_sim']*100:.0f}%")
        st.caption(f"Model: {r['model']}  ·  Embeddings: all-MiniLM-L6-v2  ·  Vector store: FAISS")
        st.download_button("⬇️  Download answer report (.txt)", make_report(r),
                           file_name="answer_report.txt", use_container_width=True)

    st.markdown('<div class="disc">⚕️ <b>Disclaimer:</b> AI research prototype for an MSc '
                'dissertation. It provides general information from a medical dataset and is '
                '<b>not a replacement for professional medical advice</b>.</div>',
                unsafe_allow_html=True)
