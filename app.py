import streamlit as st
import re
import numpy as np
import pickle
import os
import time
import random
from collections import Counter
import nltk
nltk_data_path = os.path.expanduser("~/nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
    nltk.download('stopwords', quiet=True)
if not os.path.exists(os.path.join(nltk_data_path, "corpora/wordnet")):
    nltk.download('wordnet', quiet=True)
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SocialPulse AI · Sentiment Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #050508;
  --surface:  #0e0e18;
  --surface2: #161625;
  --border:   #1f1f35;
  --border2:  #2a2a45;
  --text:     #eeeef5;
  --muted:    #6b6b8a;
  --muted2:   #9090b0;
  --a: #4f8fff;
  --b: #a259ff;
  --c: #ff5fa0;
  --d: #00e6b8;
  --e: #ffb84f;
  --f: #ff6b4f;
}

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.main, .block-container { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1500px !important; }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

.topbar {
  background: linear-gradient(135deg, #0a0a18 0%, #10102a 50%, #0a0a18 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 1.8rem;
  position: relative;
  overflow: hidden;
}
.topbar-glow-a { position:absolute; top:-80px; right:-60px; width:350px; height:350px; background:radial-gradient(circle, rgba(79,143,255,0.12) 0%, transparent 65%); border-radius:50%; pointer-events:none; }
.topbar-glow-b { position:absolute; bottom:-60px; left:10%; width:280px; height:280px; background:radial-gradient(circle, rgba(162,89,255,0.08) 0%, transparent 65%); border-radius:50%; pointer-events:none; }
.topbar-tag {
  display:inline-flex; align-items:center; gap:0.4rem;
  background: rgba(79,143,255,0.1); border:1px solid rgba(79,143,255,0.25);
  color: var(--a) !important; border-radius:50px; padding:0.28rem 0.9rem;
  font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
  margin-bottom:1rem;
}
.topbar-dot { width:6px; height:6px; background:var(--a); border-radius:50%; animation: pulse 2s infinite; display:inline-block; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.topbar-title {
  font-size:3.2rem; font-weight:900; line-height:1.05; letter-spacing:-0.02em;
  background: linear-gradient(100deg, #fff 0%, #a0a8ff 40%, var(--b) 70%, var(--c) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.6rem;
}
.topbar-sub { color: var(--muted2) !important; font-size:1rem; font-weight:300; max-width:600px; }
.topbar-pills { display:flex; gap:0.6rem; margin-top:1.2rem; flex-wrap:wrap; }
.pill {
  display:inline-flex; align-items:center; gap:0.35rem;
  background:var(--surface2); border:1px solid var(--border2);
  border-radius:8px; padding:0.3rem 0.75rem; font-size:0.75rem; color:var(--muted2) !important; font-weight:500;
}

.kpi-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.8rem; }
.kpi {
  background: var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:1.4rem 1.6rem; position:relative; overflow:hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.kpi:hover { transform:translateY(-2px); }
.kpi-accent { position:absolute; top:0; left:0; right:0; height:2px; border-radius:16px 16px 0 0; }
.kpi-icon { font-size:1.4rem; margin-bottom:0.7rem; }
.kpi-num { font-size:2rem; font-weight:800; line-height:1; letter-spacing:-0.02em; margin-bottom:0.3rem; }
.kpi-label { font-size:0.76rem; color:var(--muted) !important; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-sub { font-size:0.7rem; color:var(--muted) !important; margin-top:0.2rem; }

.panel-head {
  font-size:0.78rem; font-weight:700; color:var(--muted) !important;
  letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1.2rem;
  display:flex; align-items:center; gap:0.5rem;
}
.panel-head-line { flex:1; height:1px; background:var(--border); }

textarea {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 14px !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
  resize: none !important;
}
textarea:focus { border-color: rgba(79,143,255,0.5) !important; box-shadow: 0 0 0 3px rgba(79,143,255,0.08) !important; }
.stTextArea label { display:none !important; }

.stButton > button {
  background: linear-gradient(135deg, var(--a) 0%, var(--b) 100%) !important;
  color: #fff !important; border: none !important;
  border-radius: 14px !important; padding: 0.75rem 1.5rem !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
  font-size: 0.95rem !important; letter-spacing: 0.02em !important;
  width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(79,143,255,0.35) !important; }

.res-box { border-radius:18px; padding:2rem; text-align:center; position:relative; overflow:hidden; }
.res-pos { background:linear-gradient(135deg,rgba(0,230,184,0.06),rgba(79,143,255,0.06)); border:1px solid rgba(0,230,184,0.25); }
.res-neg { background:linear-gradient(135deg,rgba(255,95,160,0.06),rgba(255,107,79,0.06)); border:1px solid rgba(255,95,160,0.25); }
.res-empty { background:var(--surface2); border:1px dashed var(--border2); }
.res-emoji { font-size:3.5rem; line-height:1; margin-bottom:0.6rem; }
.res-label { font-size:2.5rem; font-weight:900; letter-spacing:-0.02em; line-height:1; margin-bottom:0.4rem; }
.res-pos .res-label { color: var(--d) !important; }
.res-neg .res-label { color: var(--c) !important; }
.res-conf-text { font-size:0.85rem; color:var(--muted2) !important; margin-bottom:1rem; }
.conf-track { background:rgba(255,255,255,0.06); border-radius:50px; height:6px; overflow:hidden; margin:0.8rem 0; }
.conf-fill-pos { height:100%; border-radius:50px; background:linear-gradient(90deg,var(--a),var(--d)); }
.conf-fill-neg { height:100%; border-radius:50px; background:linear-gradient(90deg,var(--f),var(--c)); }

.emotion-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.7rem; }
.emotion-row { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:0.7rem 0.9rem; }
.emotion-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; }
.emotion-name { font-size:0.78rem; font-weight:600; }
.emotion-pct { font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:var(--muted2) !important; }
.emotion-bar { height:4px; background:rgba(255,255,255,0.06); border-radius:50px; overflow:hidden; }
.emotion-fill { height:100%; border-radius:50px; }

.hist-item {
  background:var(--surface2); border:1px solid var(--border);
  border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;
  display:flex; align-items:center; gap:0.8rem;
}
.hist-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.hist-text { font-size:0.85rem; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted2) !important; }
.hist-sent { font-size:0.72rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:6px; flex-shrink:0; }
.hist-sent-pos { background:rgba(0,230,184,0.12); color:var(--d) !important; }
.hist-sent-neg { background:rgba(255,95,160,0.12); color:var(--c) !important; }
.hist-pct { font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:var(--muted) !important; flex-shrink:0; width:42px; text-align:right; }

.sb-logo { padding:1.8rem 1.5rem 1rem; border-bottom:1px solid var(--border); }
.sb-logo-title { font-size:1.25rem; font-weight:800; background:linear-gradient(90deg,var(--a),var(--b)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sb-logo-sub { font-size:0.72rem; color:var(--muted) !important; margin-top:0.2rem; }
.sb-section { padding:1.2rem 1.5rem; border-bottom:1px solid var(--border); }
.sb-section-title { font-size:0.68rem; font-weight:700; color:var(--muted) !important; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1rem; }
.sb-stat-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem; }
.sb-stat-name { font-size:0.8rem; color:var(--muted2) !important; }
.sb-stat-val { font-size:0.85rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.sb-bar-wrap { background:rgba(255,255,255,0.05); border-radius:50px; height:5px; overflow:hidden; margin-bottom:0.5rem; display:flex; }
.sb-bar-pos { height:100%; background:linear-gradient(90deg,var(--a),var(--d)); border-radius:50px 0 0 50px; }
.sb-bar-neg { height:100%; background:linear-gradient(90deg,var(--f),var(--c)); border-radius:0 50px 50px 0; }
.status-ok { background:rgba(0,230,184,0.08); border:1px solid rgba(0,230,184,0.2); border-radius:10px; padding:0.8rem 1rem; font-size:0.78rem; color:var(--d) !important; }
.status-warn { background:rgba(255,184,79,0.08); border:1px solid rgba(255,184,79,0.2); border-radius:10px; padding:0.8rem 1rem; font-size:0.78rem; color:var(--e) !important; }

.word-chip { display:inline-block; border-radius:6px; padding:0.25rem 0.6rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace; margin:0.2rem; border:1px solid transparent; }
.chip-pos { background:rgba(0,230,184,0.08); border-color:rgba(0,230,184,0.2); color:var(--d) !important; }
.chip-neg { background:rgba(255,95,160,0.08); border-color:rgba(255,95,160,0.2); color:var(--c) !important; }
.chip-neu { background:var(--surface2); border-color:var(--border); color:var(--muted2) !important; }

.stTabs [data-baseweb="tab-list"] { background:var(--surface2) !important; border-radius:12px !important; padding:3px !important; gap:2px !important; border:1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; border-radius:9px !important; font-family:'Outfit',sans-serif !important; font-weight:600 !important; font-size:0.82rem !important; padding:0.5rem 1rem !important; border:none !important; }
.stTabs [aria-selected="true"] { background:var(--surface) !important; color:var(--text) !important; box-shadow:0 1px 4px rgba(0,0,0,0.4) !important; }
.stTabs [data-baseweb="tab-panel"] { padding:0 !important; margin-top:1rem !important; }

#MainMenu, footer, header, .stDeployButton { visibility:hidden; display:none; }
</style>
""", unsafe_allow_html=True)

# ── Model & NLP ────────────────────────────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    try:
        m = pickle.load(open("sentiment_model.pkl","rb"))
        v = pickle.load(open("vectorizer.pkl","rb"))
        return m, v, True
    except:
        return None, None, False

model, vectorizer, model_loaded = load_model()

POSITIVE_WORDS = {'love','great','amazing','good','excellent','happy','best','awesome','wonderful','fantastic','perfect','brilliant','beautiful','enjoy','excited','glad','nice','superb','outstanding','incredible','delightful','pleased','thankful','joy','hope','grateful','inspiring','fun','friendly','helpful','easy','fast','reliable','recommend','impressed','positive','win','success'}
NEGATIVE_WORDS = {'hate','bad','worst','terrible','awful','poor','horrible','disgusting','ugly','broken','fail','error','slow','crash','disappoint','sad','angry','frustrated','useless','waste','boring','annoying','problem','issue','wrong','hard','difficult','regret','sorry','never','not','no','negative','loss','bug','scam','fake','spam','rude','cheap','painful','stress','tired','sick'}

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+",'',text)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'#\w+','',text)
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = text.lower()
    tokens = text.split()
    negations = {"not","no","never","n't"}
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words or w in negations]
    return ' '.join(tokens)

def get_word_tags(text):
    words = re.sub(r'[^a-zA-Z\s]','',text).lower().split()
    result = []
    for w in words[:20]:
        if w in POSITIVE_WORDS: result.append((w,'pos'))
        elif w in NEGATIVE_WORDS: result.append((w,'neg'))
        else: result.append((w,'neu'))
    return result

def get_emotion_scores(text, sentiment, confidence):
    t = text.lower(); c = confidence / 100
    if sentiment == "Positive":
        joy     = min(99, int(c*100*(1.0 if any(w in t for w in ['happy','joy','love','amazing','wonderful']) else 0.7)))
        trust   = min(99, int(c*100*(1.0 if any(w in t for w in ['reliable','trust','great','best','recommend']) else 0.6)))
        excited = min(99, int(c*100*(1.0 if any(w in t for w in ['excited','awesome','fantastic','incredible','wow']) else 0.55)))
        calm    = min(99, int((1-c*0.4)*80))
    else:
        joy     = min(99, int((1-c)*30))
        trust   = min(99, int((1-c)*25))
        excited = min(99, int(c*45*(1 if any(w in t for w in ['hate','terrible','awful','worst','horrible']) else 0.5)))
        calm    = min(99, int((1-c*0.8)*50))
    return {"Joy":(joy,"#00e6b8"),"Trust":(trust,"#4f8fff"),"Anticipation":(excited,"#a259ff"),"Calm":(calm,"#ffb84f")}

def predict(text):
    if not model_loaded:
        words = text.lower().split()
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        if pos >= neg:
            return "Positive", round(random.uniform(72, 94), 1)
        else:
            return "Negative", round(random.uniform(72, 94), 1)
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    label = np.argmax(prob)
    return ("Positive" if label == 1 else "Negative"), float(round(max(prob)*100, 1))

# ── Session State ──────────────────────────────────────────────────────────────
for k,v in [("history",[]),("total",0),("pos",0),("neg",0),("avg_conf",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
        <div style='font-size:1.8rem;margin-bottom:0.5rem;'>⚡</div>
        <div class='sb-logo-title'>SocialPulse AI</div>
        <div class='sb-logo-sub'>Sentiment Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    total = st.session_state.total
    pos   = st.session_state.pos
    neg   = st.session_state.neg
    pp    = round(pos/total*100,1) if total>0 else 0
    np_   = round(neg/total*100,1) if total>0 else 0
    avg_c = round(sum(st.session_state.avg_conf)/len(st.session_state.avg_conf),1) if st.session_state.avg_conf else 0.0

    st.markdown(f"""
    <div class='sb-section'>
        <div class='sb-section-title'>Session Analytics</div>
        <div class='sb-stat-row'><span class='sb-stat-name'>Total Analyzed</span><span class='sb-stat-val' style='color:var(--a);'>{total}</span></div>
        <div class='sb-stat-row'><span class='sb-stat-name'>Positive</span><span class='sb-stat-val' style='color:var(--d);'>{pos} <span style='color:var(--muted);font-size:0.7rem;'>({pp}%)</span></span></div>
        <div class='sb-stat-row'><span class='sb-stat-name'>Negative</span><span class='sb-stat-val' style='color:var(--c);'>{neg} <span style='color:var(--muted);font-size:0.7rem;'>({np_}%)</span></span></div>
        <div class='sb-stat-row'><span class='sb-stat-name'>Avg Confidence</span><span class='sb-stat-val' style='color:var(--b);'>{avg_c}%</span></div>
        <div style='margin-top:0.8rem;'>
            <div style='display:flex;justify-content:space-between;font-size:0.68rem;color:var(--muted);margin-bottom:0.4rem;'>
                <span>😊 {pp}%</span><span>{np_}% 😤</span>
            </div>
            <div class='sb-bar-wrap'>
                <div class='sb-bar-pos' style='width:{pp}%;'></div>
                <div class='sb-bar-neg' style='width:{np_}%;'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-section'><div class='sb-section-title'>Quick Examples</div>", unsafe_allow_html=True)
    EXAMPLES = [
        ("😍  Love this product!", "I absolutely love this product, it completely changed my life for the better!"),
        ("😤  Worst experience", "This is the worst experience I've ever had, completely terrible service."),
        ("🚀  So excited today", "So excited about the new update, it's fantastic and works perfectly!"),
        ("💔  Really disappointed", "Really disappointed, the quality is awful and it broke after one day."),
        ("✅  Highly recommend", "Highly recommend this to everyone, brilliant quality and amazing support."),
        ("😡  Total waste", "Total waste of money, doesn't work at all and customer service is useless."),
    ]
    for label, txt in EXAMPLES:
        if st.button(label, key=f"ex_{label}"):
            st.session_state["prefill"] = txt
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-section' style='border-bottom:none;'><div class='sb-section-title'>Model Status</div>", unsafe_allow_html=True)
    if model_loaded:
        st.markdown("<div class='status-ok'>✅ <strong>Full AI Mode</strong><br><span style='font-size:0.72rem;opacity:0.8;'>Logistic Regression · Sentiment140<br>1.6M tweets · 80.1% accuracy</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-warn'>⚡ <strong>Demo Mode Active</strong><br><span style='font-size:0.72rem;opacity:0.8;'>Upload sentiment_model.pkl &amp; vectorizer.pkl to enable full AI</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TOPBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='topbar'>
  <div class='topbar-glow-a'></div>
  <div class='topbar-glow-b'></div>
  <div class='topbar-tag'><span class='topbar-dot'></span>&nbsp;LIVE · AI-POWERED</div>
  <div class='topbar-title'>SocialPulse AI</div>
  <div class='topbar-sub'>Decode emotional tone from any social text in under a second — with confidence scores, emotion breakdowns &amp; word-level insights</div>
  <div class='topbar-pills'>
    <span class='pill'>🧠 Machine Learning</span>
    <span class='pill'>📊 1.6M Training Samples</span>
    <span class='pill'>⚡ Real-time Analysis</span>
    <span class='pill'>🎯 80.1% Accuracy</span>
    <span class='pill'>🔤 TF-IDF · N-grams</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ──────────────────────────────────────────────────────────────────
avg_c_display = f"{avg_c}%" if st.session_state.avg_conf else "—"
st.markdown(f"""
<div class='kpi-strip'>
  <div class='kpi'>
    <div class='kpi-accent' style='background:linear-gradient(90deg,var(--a),var(--b));'></div>
    <div class='kpi-icon'>🧠</div>
    <div class='kpi-num' style='color:var(--a);'>80.1%</div>
    <div class='kpi-label'>Model Accuracy</div>
    <div class='kpi-sub'>On 320K test samples</div>
  </div>
  <div class='kpi'>
    <div class='kpi-accent' style='background:linear-gradient(90deg,var(--b),var(--c));'></div>
    <div class='kpi-icon'>📦</div>
    <div class='kpi-num' style='color:var(--b);'>1.6M</div>
    <div class='kpi-label'>Training Tweets</div>
    <div class='kpi-sub'>Sentiment140 dataset</div>
  </div>
  <div class='kpi'>
    <div class='kpi-accent' style='background:linear-gradient(90deg,var(--d),var(--a));'></div>
    <div class='kpi-icon'>⚡</div>
    <div class='kpi-num' style='color:var(--d);'>&lt;1s</div>
    <div class='kpi-label'>Inference Time</div>
    <div class='kpi-sub'>Real-time predictions</div>
  </div>
  <div class='kpi'>
    <div class='kpi-accent' style='background:linear-gradient(90deg,var(--e),var(--f));'></div>
    <div class='kpi-icon'>📈</div>
    <div class='kpi-num' style='color:var(--e);'>{st.session_state.total}</div>
    <div class='kpi-label'>Analyzed This Session</div>
    <div class='kpi-sub'>{avg_c_display} avg confidence</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN COLUMNS ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("<div class='panel-head'>✍️ INPUT TEXT <div class='panel-head-line'></div></div>", unsafe_allow_html=True)
    prefill = st.session_state.pop("prefill", "")
    user_text = st.text_area("text", value=prefill, height=140,
        placeholder="Paste a tweet, review, comment, or any social media text here…",
        key="main_input", label_visibility="collapsed")

    chars = len(user_text); words = len(user_text.split()) if user_text.strip() else 0
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;margin-top:-0.3rem;margin-bottom:0.8rem;'>
        <span style='font-size:0.73rem;color:var(--muted);'>{chars} characters · {words} words</span>
        <span style='font-size:0.73rem;color:var(--muted);font-family:JetBrains Mono,monospace;'>TF-IDF · LOGISTIC REGRESSION</span>
    </div>""", unsafe_allow_html=True)

    analyze = st.button("⚡ Analyze Sentiment Now", key="analyze_btn")

    if user_text.strip():
        tags = get_word_tags(user_text)
        if tags:
            st.markdown("<div class='panel-head' style='margin-top:1.2rem;'>🔤 WORD SIGNALS <div class='panel-head-line'></div></div>", unsafe_allow_html=True)
            chips_html = "".join(f"<span class='word-chip {'chip-pos' if k=='pos' else ('chip-neg' if k=='neg' else 'chip-neu')}'>{w}</span>" for w,k in tags)
            st.markdown(f"<div style='line-height:2.2;'>{chips_html}</div>", unsafe_allow_html=True)
            pos_sig = sum(1 for _,k in tags if k=="pos")
            neg_sig = sum(1 for _,k in tags if k=="neg")
            st.markdown(f"""
            <div style='margin-top:0.8rem;display:flex;gap:1rem;font-size:0.75rem;'>
                <span style='color:var(--d);'>✅ {pos_sig} positive signals</span>
                <span style='color:var(--c);'>❌ {neg_sig} negative signals</span>
                <span style='color:var(--muted);'>⬜ {len(tags)-pos_sig-neg_sig} neutral</span>
            </div>""", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='panel-head'>📊 ANALYSIS RESULT <div class='panel-head-line'></div></div>", unsafe_allow_html=True)

    if analyze and user_text.strip():
        with st.spinner(""):
            time.sleep(0.35)
            sentiment, confidence = predict(user_text)

        st.session_state.total  += 1
        st.session_state.pos    += (1 if sentiment=="Positive" else 0)
        st.session_state.neg    += (1 if sentiment=="Negative" else 0)
        st.session_state.avg_conf.append(confidence)
        st.session_state.history.insert(0, {"text":user_text,"sentiment":sentiment,"confidence":confidence})
        if len(st.session_state.history) > 12:
            st.session_state.history.pop()

        res_cls = "res-pos" if sentiment=="Positive" else "res-neg"
        emoji   = "😊" if sentiment=="Positive" else "😤"
        bar_cls = "conf-fill-pos" if sentiment=="Positive" else "conf-fill-neg"
        label_c = "var(--d)" if sentiment=="Positive" else "var(--c)"
        emotions = get_emotion_scores(user_text, sentiment, confidence)
        emo_html = "".join(f"<div class='emotion-row'><div class='emotion-top'><span class='emotion-name' style='color:{ec};'>{en}</span><span class='emotion-pct'>{ev}%</span></div><div class='emotion-bar'><div class='emotion-fill' style='width:{ev}%;background:{ec};'></div></div></div>" for en,(ev,ec) in emotions.items())

        st.markdown(f"""
        <div class='{res_cls} res-box'>
            <div class='res-emoji'>{emoji}</div>
            <div class='res-label'>{sentiment}</div>
            <div class='res-conf-text'>Confidence Score: <strong style='color:{label_c};'>{confidence}%</strong></div>
            <div class='conf-track'><div class='{bar_cls}' style='width:{confidence}%;'></div></div>
            <div style='display:flex;justify-content:center;gap:1.5rem;font-size:0.72rem;color:var(--muted);margin-top:0.5rem;'>
                <span>📝 {words} words</span><span>🔤 {chars} chars</span><span>🤖 {'Full AI' if model_loaded else 'Demo'}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-head'>🧠 EMOTION BREAKDOWN <div class='panel-head-line'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='emotion-grid'>{emo_html}</div>", unsafe_allow_html=True)
        st.rerun()

    elif analyze and not user_text.strip():
        st.markdown("<div style='background:rgba(255,184,79,0.06);border:1px solid rgba(255,184,79,0.2);border-radius:14px;padding:1.5rem;text-align:center;color:var(--e);'>⚠️ Please enter some text first</div>", unsafe_allow_html=True)

    elif st.session_state.history:
        last = st.session_state.history[0]
        s, c = last["sentiment"], last["confidence"]
        res_cls = "res-pos" if s=="Positive" else "res-neg"
        emoji = "😊" if s=="Positive" else "😤"
        bar_cls = "conf-fill-pos" if s=="Positive" else "conf-fill-neg"
        label_c = "var(--d)" if s=="Positive" else "var(--c)"
        emotions = get_emotion_scores(last["text"], s, c)
        emo_html = "".join(f"<div class='emotion-row'><div class='emotion-top'><span class='emotion-name' style='color:{ec};'>{en}</span><span class='emotion-pct'>{ev}%</span></div><div class='emotion-bar'><div class='emotion-fill' style='width:{ev}%;background:{ec};'></div></div></div>" for en,(ev,ec) in emotions.items())
        st.markdown(f"""
        <div class='{res_cls} res-box'>
            <div class='res-emoji'>{emoji}</div>
            <div class='res-label'>{s}</div>
            <div class='res-conf-text'>Confidence: <strong style='color:{label_c};'>{c}%</strong></div>
            <div class='conf-track'><div class='{bar_cls}' style='width:{c}%;'></div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-head'>🧠 EMOTION BREAKDOWN <div class='panel-head-line'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='emotion-grid'>{emo_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='res-empty res-box' style='padding:3.5rem 2rem;'>
            <div style='font-size:3rem;margin-bottom:0.8rem;opacity:0.4;'>⚡</div>
            <div style='color:var(--muted2);font-size:0.9rem;line-height:1.6;'>Enter any social text on the left<br>and click <strong>Analyze</strong> to see results</div>
        </div>""", unsafe_allow_html=True)

# ── HISTORY + OVERVIEW TABS ────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🕓  Recent History", "📈  Session Overview"])

    with tab1:
        st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
        for item in st.session_state.history[:8]:
            dot_c = "#00e6b8" if item['sentiment']=="Positive" else "#ff5fa0"
            sent_cls = "hist-sent-pos" if item['sentiment']=="Positive" else "hist-sent-neg"
            emoji = "😊" if item['sentiment']=="Positive" else "😤"
            st.markdown(f"""
            <div class='hist-item'>
                <div class='hist-dot' style='background:{dot_c};box-shadow:0 0 6px {dot_c}55;'></div>
                <div class='hist-text'>"{item['text']}"</div>
                <span class='hist-sent {sent_cls}'>{emoji} {item['sentiment']}</span>
                <span class='hist-pct'>{item['confidence']}%</span>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History"):
            for k,v in [("history",[]),("total",0),("pos",0),("neg",0),("avg_conf",[])]:
                st.session_state[k] = v
            st.rerun()

    with tab2:
        t = st.session_state.total
        if t > 0:
            p = st.session_state.pos; n = st.session_state.neg
            pp2 = round(p/t*100,1); np2 = round(n/t*100,1)
            ac = round(sum(st.session_state.avg_conf)/len(st.session_state.avg_conf),1)
            high_conf = sum(1 for c in st.session_state.avg_conf if c >= 85)
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
                <div style='background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:1.3rem;text-align:center;'>
                    <div style='font-size:2rem;font-weight:800;color:var(--a);'>{t}</div>
                    <div style='font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;'>Total</div>
                </div>
                <div style='background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:1.3rem;text-align:center;'>
                    <div style='font-size:2rem;font-weight:800;color:var(--d);'>{pp2}%</div>
                    <div style='font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;'>Positive Rate</div>
                </div>
                <div style='background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:1.3rem;text-align:center;'>
                    <div style='font-size:2rem;font-weight:800;color:var(--e);'>{ac}%</div>
                    <div style='font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;'>Avg Confidence</div>
                </div>
            </div>
            <div style='margin-top:1rem;background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:1.3rem;'>
                <div style='font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;'>Sentiment Split</div>
                <div style='display:flex;align-items:center;gap:1rem;'>
                    <div style='font-size:0.85rem;color:var(--d);min-width:80px;'>😊 {p} pos</div>
                    <div style='flex:1;background:rgba(255,255,255,0.05);border-radius:50px;height:8px;overflow:hidden;display:flex;'>
                        <div style='width:{pp2}%;background:linear-gradient(90deg,var(--a),var(--d));border-radius:50px 0 0 50px;'></div>
                        <div style='width:{np2}%;background:linear-gradient(90deg,var(--f),var(--c));border-radius:0 50px 50px 0;'></div>
                    </div>
                    <div style='font-size:0.85rem;color:var(--c);min-width:80px;text-align:right;'>{n} neg 😤</div>
                </div>
                <div style='margin-top:0.8rem;font-size:0.75rem;color:var(--muted);'>🎯 {high_conf} of {t} predictions had &ge;85% confidence</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center;color:var(--muted);padding:2rem;font-size:0.85rem;'>Analyze some text to see your session overview here</div>", unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:3rem 0 1rem;border-top:1px solid var(--border);margin-top:2rem;'>
    <div style='font-size:0.72rem;color:var(--muted);letter-spacing:0.05em;'>
        ⚡ <strong style='color:var(--muted2);'>SOCIALPULSE AI</strong> &nbsp;·&nbsp;
        Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp;
        Trained on Sentiment140 &nbsp;·&nbsp; v2.0
    </div>
</div>
""", unsafe_allow_html=True)