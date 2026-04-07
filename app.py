#╔══════════════════════════════════════════════════════════════════════════╗
# ║           MOODTUNES – AI Mood-Based Music Recommender                   ║                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import json
import datetime
import hashlib
import requests
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image

# ── Optional DeepFace (graceful fallback if not installed) ──────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="MoodTunes – AI Music Recommender",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="auto",
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL CSS – White Theme                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#ffffff; color:#1a1a2e; }
.stApp { background: #ffffff; }
h1,h2,h3 { font-family:'Orbitron',monospace; color:#1a1a2e; }

[data-testid="stSidebar"] {
    background: #f0f4f8 !important;
    border-right: 1px solid #d0d8e4;
}
.metric-card {
    background: #f7f9fc;
    border:1px solid #b0c4de; border-radius:12px;
    padding:18px 22px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:12px;
}
.metric-card .val { font-family:'Orbitron',monospace; font-size:2rem; color:#0077aa; }
.metric-card .lbl { font-size:.8rem; color:#555577; margin-top:4px; text-transform:uppercase; letter-spacing:.08em; }

.song-card {
    background: #f7f9fc;
    border:1px solid #b0c4de; border-radius:14px;
    padding:20px 24px; margin-bottom:16px; transition:all .25s;
}
.song-card:hover { border-color:#0077aa; box-shadow:0 4px 16px rgba(0,119,170,.12); }
.song-title { font-family:'Orbitron',monospace; font-size:1.05rem; color:#0077aa; }
.song-artist { color:#228855; font-size:.9rem; margin-top:2px; }
.emotion-badge {
    display:inline-block; background:#e8f0fe; border:1px solid #0077aa55;
    border-radius:20px; padding:3px 12px; font-size:.75rem; color:#0077aa; margin-top:8px;
}
.match-bar-bg { background:#e0e8f0; border-radius:4px; height:6px; margin-top:8px; }
.match-bar-fill { background:linear-gradient(90deg,#0077aa,#00bb77); border-radius:4px; height:6px; }

.mood-display {
    background: #f0f7ff;
    border:1px solid #0077aa44; border-radius:16px;
    padding:24px; text-align:center; margin-bottom:20px;
}
.mood-emoji { font-size:3.5rem; }
.mood-label { font-family:'Orbitron',monospace; font-size:1.4rem; color:#0077aa; margin-top:8px; }
.mood-score { color:#228855; font-size:.9rem; margin-top:4px; }

.stButton > button {
    background: linear-gradient(90deg,#0077aa,#00aa77) !important;
    color:#ffffff !important; font-family:'Orbitron',monospace !important;
    font-weight:700 !important; border:none !important; border-radius:8px !important;
    padding:10px 24px !important; letter-spacing:.05em !important; transition:all .2s !important;
}
.stButton > button:hover { opacity:.85; box-shadow:0 4px 14px rgba(0,119,170,.3) !important; }

.stTabs [data-baseweb="tab-list"] { background:#f0f4f8; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#555577 !important; font-family:'Orbitron',monospace !important; font-size:.75rem; }
.stTabs [aria-selected="true"] { background:#ffffff !important; color:#0077aa !important; border-radius:7px; box-shadow:0 1px 4px rgba(0,0,0,.1); }

.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background:#ffffff !important; color:#1a1a2e !important;
    border:1px solid #b0c4de !important; border-radius:8px !important;
}
hr { border-color:#d0d8e4; }
.pl-item {
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 14px; background:#f7f9fc; border:1px solid #d0d8e4;
    border-radius:8px; margin-bottom:8px;
}
.pl-song { font-size:.9rem; color:#1a1a2e; }
.pl-artist { font-size:.78rem; color:#555577; }
.history-item { border-left:2px solid #0077aa44; padding:8px 16px; margin-bottom:10px; }
.history-time { font-size:.75rem; color:#888899; }
.history-mood { color:#228855; font-size:.85rem; font-weight:600; }
.stProgress > div > div { background:linear-gradient(90deg,#0077aa,#00aa77) !important; }

.upload-dropzone {
    border: 2px dashed #b0c4de; border-radius: 14px; padding: 48px 24px;
    text-align: center; color: #555577; margin-top: 12px;
    transition: border-color .25s; background:#f7f9fc;
}
.upload-dropzone:hover { border-color: #0077aa; }
.upload-dropzone-icon { font-size: 2.5rem; margin-bottom: 12px; }
.upload-dropzone-title { font-family: Orbitron, monospace; color: #0077aa; font-size: .95rem; }
.upload-dropzone-sub { font-size: .8rem; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FILE PATHS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
USER_FILE     = "users.csv"
PLAYLIST_FILE = "playlists.json"
HISTORY_FILE  = "history.json"
FEEDBACK_FILE = "feedback.json"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  JSON PERSISTENCE HELPERS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  USER AUTH  (original login_page logic – upgraded with hashing)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username","password","created_at","avatar_color"]).to_csv(USER_FILE, index=False)

def load_users():
    return pd.read_csv(USER_FILE)

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False, "Username already taken."
    colors = ["#0077aa","#228855","#cc4477","#dd8800","#7744bb"]
    color  = colors[len(df) % len(colors)]
    new_row = pd.DataFrame(
        [[username, _hash(password), datetime.datetime.now().isoformat(), color]],
        columns=["username","password","created_at","avatar_color"]
    )
    pd.concat([df, new_row], ignore_index=True).to_csv(USER_FILE, index=False)
    return True, "Account created successfully!"

def verify_user(username, password):
    df = load_users()
    hashed = _hash(password)
    match = df[
        (df["username"] == username) &
        ((df["password"] == hashed) | (df["password"] == password))
    ]
    return not match.empty

def get_user_color(username):
    df = load_users()
    row = df[df["username"] == username]
    if not row.empty and "avatar_color" in row.columns:
        return row.iloc[0]["avatar_color"]
    return "#0077aa"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PLAYLIST MANAGEMENT                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def get_user_playlists(username):
    return _load_json(PLAYLIST_FILE, {}).get(username, {})

def save_playlist(username, pl_name, songs):
    data = _load_json(PLAYLIST_FILE, {})
    data.setdefault(username, {})[pl_name] = songs
    _save_json(PLAYLIST_FILE, data)

def delete_playlist(username, pl_name):
    data = _load_json(PLAYLIST_FILE, {})
    if username in data and pl_name in data[username]:
        del data[username][pl_name]
        _save_json(PLAYLIST_FILE, data)

def add_song_to_playlist(username, pl_name, song_dict):
    data = _load_json(PLAYLIST_FILE, {})
    data.setdefault(username, {}).setdefault(pl_name, [])
    if song_dict["song"] not in [s["song"] for s in data[username][pl_name]]:
        data[username][pl_name].append(song_dict)
        _save_json(PLAYLIST_FILE, data)
        return True
    return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION HISTORY                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def log_history(username, mood, emotion, songs_count, input_method):
    data = _load_json(HISTORY_FILE, {})
    data.setdefault(username, []).append({
        "timestamp": datetime.datetime.now().isoformat(),
        "mood": mood, "emotion": emotion,
        "songs_recommended": songs_count, "input_method": input_method,
    })
    data[username] = data[username][-100:]
    _save_json(HISTORY_FILE, data)

def get_history(username):
    return _load_json(HISTORY_FILE, {}).get(username, [])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FEEDBACK / RATINGS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def save_feedback(username, song, artist, rating, comment=""):
    data = _load_json(FEEDBACK_FILE, {})
    data.setdefault(username, []).append({
        "timestamp": datetime.datetime.now().isoformat(),
        "song": song, "artist": artist, "rating": rating, "comment": comment
    })
    _save_json(FEEDBACK_FILE, data)

def get_feedback(username):
    return _load_json(FEEDBACK_FILE, {}).get(username, [])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING & PREPROCESSING                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@st.cache_resource
def load_data():
    df = pd.read_csv("spotify_millsongdata.csv")
    if len(df) > 8000:
        df = df.sample(8000, random_state=42).reset_index(drop=True)

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        return text.lower().strip()

    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = df["text"].apply(preprocess_text)
    else:
        df["cleaned_text"] = df["cleaned_text"].fillna(df["text"].apply(preprocess_text))

    def get_textblob_sentiment(text):
        try:    return TextBlob(text).sentiment.polarity
        except: return 0.0

    _v = SentimentIntensityAnalyzer()
    def get_vader_sentiment(text):
        try:    return _v.polarity_scores(text)["compound"]
        except: return 0.0

    df["sentiment_score"] = df["cleaned_text"].apply(get_vader_sentiment)
    df["textblob_score"]  = df["cleaned_text"].apply(get_textblob_sentiment)

    def classify_emotion(score):
        if score >  0.5:  return "Happy / Energetic"
        if score >  0.1:  return "Romantic / Positive"
        if score > -0.1:  return "Neutral / Calm"
        if score > -0.5:  return "Sad"
        return "Deep Sad"

    df["emotion"] = df["sentiment_score"].apply(classify_emotion)

    if "genre" not in df.columns:
        genres = ["Pop","Rock","R&B","Hip-Hop","Classical","Jazz","Electronic","Country","Folk","Latin"]
        df["genre"] = np.random.default_rng(42).choice(genres, size=len(df))

    tfidf        = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"].fillna(""))
    cosine_sim   = cosine_similarity(tfidf_matrix)

    return df, tfidf, tfidf_matrix, cosine_sim


df, tfidf, tfidf_matrix, cosine_sim = load_data()
analyzer = SentimentIntensityAnalyzer()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EMOTION META MAP                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
EMOTION_MAP = {
    "Happy / Energetic":   {"emoji":"😄","color":"#FFD700"},
    "Romantic / Positive": {"emoji":"💖","color":"#FF69B4"},
    "Neutral / Calm":      {"emoji":"😌","color":"#87CEEB"},
    "Sad":                 {"emoji":"😢","color":"#6495ED"},
    "Deep Sad":            {"emoji":"💔","color":"#8B5CF6"},
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MOOD DETECTION FUNCTIONS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def detect_user_mood_from_text(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.6:  return score, "Happy",    "Energetic",    "Happy / Energetic"
    if score > 0.2:  return score, "Positive", "Romantic",     "Romantic / Positive"
    if score > -0.2: return score, "Neutral",  "Calm",         "Neutral / Calm"
    if score > -0.6: return score, "Sad",      "Calm",         "Sad"
    return score, "Very Sad", "Motivational", "Deep Sad"


def map_face_emotion_to_label(face_emotion):
    e = str(face_emotion).lower()
    if "happy"    in e or "joy"     in e: return  0.8, "Happy",     "Energetic",    "Happy / Energetic"
    if "sad"      in e or "disgust" in e: return -0.5, "Sad",       "Calm",         "Sad"
    if "angry"    in e:                   return -0.6, "Angry",     "Calm",         "Sad"
    if "surprise" in e:                   return  0.2, "Surprised", "Energetic",    "Happy / Energetic"
    if "fear"     in e:                   return -0.7, "Fearful",   "Calm",         "Deep Sad"
    return 0.0, "Neutral", "Calm", "Neutral / Calm"


def detect_face_emotion_from_image(pil_image):
    if not DEEPFACE_AVAILABLE:
        return None
    try:
        img    = np.array(pil_image.convert("RGB"))
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list): return result[0].get("dominant_emotion", None)
        return result.get("dominant_emotion", None)
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None


# ── Weather → Mood ──────────────────────────────────────────────────────────
def weather_to_mood(weather_desc):
    w = weather_desc.lower()
    if any(k in w for k in ["clear","sun"]):             return  0.7, "Happy / Energetic"
    if any(k in w for k in ["cloud","overcast"]):        return  0.0, "Neutral / Calm"
    if any(k in w for k in ["rain","drizzle","shower"]): return -0.4, "Sad"
    if any(k in w for k in ["storm","thunder","snow"]):  return -0.7, "Deep Sad"
    if any(k in w for k in ["fog","mist","haze"]):       return -0.1, "Neutral / Calm"
    return 0.0, "Neutral / Calm"

def _wmo_code(code):
    if code == 0:             return "Clear sky"
    if code in [1,2,3]:       return "Cloudy"
    if code in range(51,68):  return "Rainy"
    if code in range(71,78):  return "Snowy"
    if code in range(80,83):  return "Rain showers"
    if code in range(95,100): return "Thunderstorm"
    return "Cloudy"

def get_weather(city):
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1", timeout=5
        ).json()
        if not geo.get("results"): return None, None
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        wx  = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true&forecast_days=1", timeout=5
        ).json()
        code = wx.get("current_weather", {}).get("weathercode", 0)
        temp = wx.get("current_weather", {}).get("temperature", "?")
        return _wmo_code(code), temp
    except Exception:
        return None, None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RECOMMENDATION ENGINE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def recommend_by_emotion_label(emotion_label, input_score, user_text,
                                recommend_type, user_feeling, top_n=10,
                                genre_filter=None, diversity=False):
    matching_songs = df[df["emotion"] == emotion_label].copy()
    if matching_songs.empty:
        return None, f"No songs found for emotion: {emotion_label}"

    if genre_filter and genre_filter != "All":
        filtered = matching_songs[matching_songs["genre"] == genre_filter]
        if not filtered.empty:
            matching_songs = filtered

    matching_songs["sentiment_diff"] = abs(matching_songs["sentiment_score"] - input_score)
    matching_songs = matching_songs.sort_values("sentiment_diff")

    if diversity:
        seen, rows = set(), []
        for _, row in matching_songs.iterrows():
            if row["artist"] not in seen:
                rows.append(row); seen.add(row["artist"])
            if len(rows) >= top_n: break
        matching_songs = pd.DataFrame(rows)
    else:
        matching_songs = matching_songs.head(top_n)

    results = []
    for _, row in matching_songs.iterrows():
        match_pct = min(100, max(0, round((1 - abs(row["sentiment_score"] - input_score) / 2) * 100)))
        results.append({
            "song":    row["song"],
            "artist":  row["artist"],
            "emotion": row["emotion"],
            "genre":   row.get("genre","—"),
            "match":   match_pct,
            "sentiment_score": row["sentiment_score"],
            "explanation": (
                f"Because you said: *\"{user_text}\"*\n\n"
                f"Detected mood: **{user_feeling}** · "
                f"Recommend type: **{recommend_type}** · "
                f"Sentiment closeness: {1 - abs(row['sentiment_score'] - input_score):.2f}"
            ),
        })
    return results, None


def get_recommendations(user_text, camera_image, prefer_camera=False,
                        top_n=10, genre_filter=None, diversity=False):
    if user_text and user_text.strip() != "" and not prefer_camera:
        score, feeling, rtype, emotion = detect_user_mood_from_text(user_text)
        return recommend_by_emotion_label(emotion, score, user_text, rtype, feeling,
                                          top_n=top_n, genre_filter=genre_filter, diversity=diversity)
    if camera_image is not None:
        try:    pil_image = Image.open(io.BytesIO(camera_image.getvalue()))
        except Exception:
            try:    pil_image = Image.open(camera_image)
            except: return None, "Couldn't read image from camera input."
        dominant = detect_face_emotion_from_image(pil_image)
        if dominant is None:
            return None, "Could not detect face emotion from the image."
        score, feeling, rtype, emotion = map_face_emotion_to_label(dominant)
        return recommend_by_emotion_label(emotion, score, f"Face: {dominant}", rtype, feeling,
                                          top_n=top_n, genre_filter=genre_filter, diversity=diversity)
    return None, "Please type how you feel or allow camera input."


# ── Content-based lyric similarity ──────────────────────────────────────────
def content_similar(song_name, top_n=6):
    idx_list = df.index[df["song"].str.lower() == song_name.lower()].tolist()
    if not idx_list: return []
    idx        = idx_list[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [{"song": df.iloc[i]["song"], "artist": df.iloc[i]["artist"],
             "emotion": df.iloc[i]["emotion"], "genre": df.iloc[i].get("genre","—"),
             "similarity": round(score*100)} for i, score in sim_scores]


# ── Analytics helper ────────────────────────────────────────────────────────
def compute_stats(username):
    history   = get_history(username)
    playlists = get_user_playlists(username)
    top_e     = Counter(h["emotion"] for h in history).most_common(1)
    return (len(history),
            sum(len(v) for v in playlists.values()),
            top_e[0][0] if top_e else "—",
            len(playlists))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UI COMPONENTS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def mood_card(emotion_label, score, label, method):
    meta = EMOTION_MAP.get(emotion_label, {"emoji":"🎵","color":"#0077aa"})
    st.markdown(f"""
    <div class='mood-display'>
        <div class='mood-emoji'>{meta['emoji']}</div>
        <div class='mood-label'>{emotion_label}</div>
        <div class='mood-score'>Detected: <b>{label}</b> &nbsp;·&nbsp; Score: {score:.2f} &nbsp;·&nbsp; Via: {method}</div>
    </div>
    """, unsafe_allow_html=True)


def song_card(r, username, idx, show_add=True):
    meta   = EMOTION_MAP.get(r["emotion"], {"emoji":"🎵"})
    match  = r.get("match", 80)
    q      = urllib.parse.quote_plus(f"{r['song']} {r['artist']}")
    sp_url = f"https://open.spotify.com/search/{q}"
    yt_url = f"https://www.youtube.com/results?search_query={q}"

    st.markdown(f"""
    <div class='song-card'>
        <div class='song-title'>{meta['emoji']} {r['song']}</div>
        <div class='song-artist'>🎤 {r['artist']}</div>
        <div>
            <span class='emotion-badge'>{r['emotion']}</span>
            <span class='emotion-badge' style='margin-left:6px;border-color:#22885544;color:#228855'>{r.get('genre','—')}</span>
        </div>
        <div style='margin-top:10px;font-size:.78rem;color:#888899'>Mood Match</div>
        <div class='match-bar-bg'><div class='match-bar-fill' style='width:{match}%'></div></div>
        <div style='font-size:.78rem;color:#228855;margin-top:4px'>{match}% match</div>
        <div style='margin-top:12px'>
            <a href='{sp_url}' target='_blank'
               style='background:#1DB954;color:#000;border-radius:6px;padding:5px 12px;
                      font-size:.78rem;text-decoration:none;margin-right:8px;font-weight:600'>▶ Spotify</a>
            <a href='{yt_url}' target='_blank'
               style='background:#FF0000;color:#fff;border-radius:6px;padding:5px 12px;
                      font-size:.78rem;text-decoration:none;font-weight:600'>▶ YouTube</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_add:
        with st.expander(f"➕ Save / Rate — {r['song']}", expanded=False):
            playlists = get_user_playlists(username)
            pl_names  = list(playlists.keys())
            col_a, col_b = st.columns(2)
            with col_a:
                if pl_names:
                    chosen = st.selectbox("Add to Playlist", pl_names, key=f"pls_{idx}")
                    if st.button("Add ➕", key=f"add_{idx}"):
                        added = add_song_to_playlist(username, chosen,
                            {"song":r["song"],"artist":r["artist"],"emotion":r["emotion"]})
                        st.success("Added! ✅" if added else "Already in playlist.")
                else:
                    st.info("Create a playlist in the 📋 Playlists tab first.")
            with col_b:
                rating  = st.slider("Rate ⭐", 1, 5, 3, key=f"rate_{idx}")
                comment = st.text_input("Comment", key=f"cmt_{idx}")
                if st.button("Submit Rating", key=f"rat_{idx}"):
                    save_feedback(username, r["song"], r["artist"], rating, comment)
                    st.success("Thanks! 🙏")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOGIN PAGE                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def login_page():
    st.markdown("""
    <div style='text-align:center;padding:48px 0 24px'>
        <div style='font-family:Orbitron,monospace;font-size:3rem;color:#0077aa;
                    letter-spacing:.1em;text-shadow:0 2px 12px rgba(0,119,170,.2)'>
            🎧 MOODTUNES
        </div>
        <div style='color:#555577;margin-top:10px;font-size:1.05rem'>
            AI-Powered Emotion Music Recommender
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([1,2,1])
    with c:
        tab_login, tab_signup = st.tabs(["🔑 Login", "🆕 Sign Up"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="li_u")
            password = st.text_input("Password", type="password", key="li_p")
            if st.button("Login →", use_container_width=True):
                if verify_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = username
                    st.success("Welcome back! 🎉")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("Choose Username", key="su_u")
            new_pass = st.text_input("Choose Password", type="password", key="su_p")
            if st.button("Create Account →", use_container_width=True):
                ok, msg = save_user(new_user, new_pass)
                (st.success if ok else st.error)(msg)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def render_sidebar(username):
    color = get_user_color(username)
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:16px 0 20px'>
        <div style='width:60px;height:60px;border-radius:50%;background:{color};
                    margin:0 auto 10px;display:flex;align-items:center;justify-content:center;
                    font-family:Orbitron,monospace;font-size:1.5rem;color:#ffffff;font-weight:900;
                    box-shadow:0 2px 10px {color}55'>{username[0].upper()}</div>
        <div style='font-family:Orbitron,monospace;color:{color};font-size:.9rem'>{username}</div>
    </div>
    """, unsafe_allow_html=True)

    sessions, saved, top_e, num_pl = compute_stats(username)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class='metric-card'><div class='val'>{sessions}</div><div class='lbl'>Sessions</div></div>
    <div class='metric-card'><div class='val'>{saved}</div><div class='lbl'>Saved Songs</div></div>
    <div class='metric-card'><div class='val'>{num_pl}</div><div class='lbl'>Playlists</div></div>
    """, unsafe_allow_html=True)
    top_emoji = EMOTION_MAP.get(top_e, {}).get("emoji","🎵")
    st.sidebar.markdown(f"**Favourite Mood:** {top_emoji} {top_e}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
    st.sidebar.caption("MoodTunes v2.0 · Major Project")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN APP  (5 Tabs)                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main_app(username):
    render_sidebar(username)

    st.markdown("""
    <div style='margin-bottom:24px'>
        <span style='font-family:Orbitron,monospace;font-size:1.9rem;color:#0077aa'>🎧 MOODTUNES</span>
        <span style='color:#555577;font-size:.9rem;margin-left:14px'>
            AI Music Recommender · Personalised · Explainable
        </span>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🎵 Discover","📋 Playlists","📊 Analytics","🔍 Explore","⚙️ Settings"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 – DISCOVER
    # ══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Detect Your Mood & Get Songs")
        input_method = st.radio(
            "Detection Method",
            ["✍️ Type Your Mood", "📷 Face Camera", "🌦️ My City Weather", "📂 Upload Image"],
            horizontal=True,
        )
        st.markdown("---")

        with st.expander("🎛️ Advanced Options", expanded=False):
            a1, a2, a3 = st.columns(3)
            with a1:
                genre_filter = st.selectbox("Genre Filter",
                    ["All","Pop","Rock","R&B","Hip-Hop","Classical","Jazz","Electronic","Country","Folk","Latin"])
            with a2:
                top_n = st.slider("Songs to Recommend", 5, 20, 10)
            with a3:
                diversity = st.checkbox("Artist Diversity Mode", value=False,
                    help="Pick songs from different artists")

        recs = []

        # ── TEXT ────────────────────────────────────────────────────────
        if "✍️" in input_method:
            col1, col2 = st.columns(2)
            with col1:
                user_input    = st.text_area("💬 Describe your mood or how your day was:",
                    placeholder="e.g. I'm feeling really stressed and tired today...", height=120)
                prefer_camera = st.checkbox("Prefer camera emotion over typed mood?", value=False)
            with col2:
                st.markdown("📷 Capture face (optional)")
                camera_image = st.camera_input(
                    "Use your webcam to capture your face (press Capture)", key="cam1")
                if camera_image:
                    st.image(camera_image, caption="Captured image", use_column_width=True)

            if st.button("🔮 Recommend Songs", use_container_width=True):
                with st.spinner("Analysing mood and finding songs…"):
                    recs, error = get_recommendations(user_input, camera_image,
                        prefer_camera=prefer_camera, top_n=top_n,
                        genre_filter=genre_filter, diversity=diversity)
                if error:
                    st.error(error)
                else:
                    if user_input.strip() and not prefer_camera:
                        sc, fl, _, em = detect_user_mood_from_text(user_input)
                        mood_card(em, sc, fl, "Text (VADER)")
                        log_history(username, fl, em, top_n, "text")
                    elif camera_image:
                        pil = Image.open(io.BytesIO(camera_image.getvalue()))
                        dom = detect_face_emotion_from_image(pil)
                        if dom:
                            sc, fl, _, em = map_face_emotion_to_label(dom)
                            mood_card(em, sc, f"Face: {dom}", "Facial Expression")
                            log_history(username, fl, em, top_n, "face")

        # ── CAMERA ──────────────────────────────────────────────────────
        elif "📷" in input_method:
            if not DEEPFACE_AVAILABLE:
                st.warning("⚠️ DeepFace not installed. Run: `pip install deepface tf-keras`")
            camera_image = st.camera_input("📷 Capture your face", key="cam2")
            if camera_image:
                st.image(camera_image, caption="Captured image", use_column_width=True)

            if st.button("🔮 Analyse My Face", use_container_width=True):
                if camera_image:
                    with st.spinner("Detecting facial emotion…"):
                        pil = Image.open(io.BytesIO(camera_image.getvalue()))
                        dom = detect_face_emotion_from_image(pil)
                    if dom:
                        sc, fl, rt, em = map_face_emotion_to_label(dom)
                        mood_card(em, sc, f"Face: {dom}", "DeepFace")
                        log_history(username, fl, em, top_n, "face")
                        recs, err = recommend_by_emotion_label(em, sc, f"Face: {dom}", rt, fl,
                            top_n=top_n, genre_filter=genre_filter, diversity=diversity)
                        if err: st.error(err)
                    else:
                        st.error("Could not detect face. Try better lighting and hold still.")
                else:
                    st.warning("Please capture a photo first.")

        # ── WEATHER ─────────────────────────────────────────────────────
        elif "🌦️" in input_method:
            city = st.text_input("🏙️ Enter your city", placeholder="Mumbai, Delhi, London…")
            if st.button("🔮 Mood from My Weather", use_container_width=True):
                if city.strip():
                    with st.spinner(f"Fetching weather for {city}…"):
                        desc, temp = get_weather(city)
                    if desc:
                        st.info(f"🌡️ **{city}**: {desc} · {temp}°C")
                        sc, em = weather_to_mood(desc)
                        mood_card(em, sc, desc, "Weather (Open-Meteo)")
                        log_history(username, desc, em, top_n, "weather")
                        recs, err = recommend_by_emotion_label(em, sc, f"Weather: {desc}",
                            "Mood-based", desc, top_n=top_n,
                            genre_filter=genre_filter, diversity=diversity)
                        if err: st.error(err)
                    else:
                        st.error("Couldn't fetch weather. Check city name.")
                else:
                    st.warning("Please enter a city name.")

        # ── UPLOAD IMAGE (NEW) ───────────────────────────────────────────
        elif "📂" in input_method:
            if not DEEPFACE_AVAILABLE:
                st.warning("⚠️ DeepFace not installed. Run: `pip install deepface tf-keras`")

            st.markdown("#### 📂 Drag & Drop or Browse an Image")
            st.caption("Upload a photo of a face — DeepFace will detect the emotion from it.")

            uploaded_file = st.file_uploader(
                "Drop your image here or click to browse",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file)
                col_prev, col_info = st.columns([1, 1])
                with col_prev:
                    st.image(pil_image, caption="Uploaded Image", use_column_width=True)
                with col_info:
                    st.markdown(f"""
**File:** {uploaded_file.name}
**Size:** {uploaded_file.size / 1024:.1f} KB
**Dimensions:** {pil_image.width} × {pil_image.height} px
                    """)

                if st.button("🔮 Detect Emotion from Image", use_container_width=True):
                    with st.spinner("Analysing face emotion from uploaded image…"):
                        dom = detect_face_emotion_from_image(pil_image)
                    if dom:
                        sc, fl, rt, em = map_face_emotion_to_label(dom)
                        mood_card(em, sc, f"Face: {dom}", "DeepFace (Upload)")
                        log_history(username, fl, em, top_n, "upload")
                        recs, err = recommend_by_emotion_label(
                            em, sc, f"Face: {dom}", rt, fl,
                            top_n=top_n, genre_filter=genre_filter, diversity=diversity,
                        )
                        if err:
                            st.error(err)
                    else:
                        st.error("Could not detect a face. Try a clearer, well-lit photo with a visible face.")
            else:
                st.markdown("""
                <div class='upload-dropzone'>
                    <div class='upload-dropzone-icon'>📂</div>
                    <div class='upload-dropzone-title'>Drag &amp; Drop your image here</div>
                    <div class='upload-dropzone-sub'>Supports JPG, JPEG, PNG, WEBP &nbsp;·&nbsp; Best results with a clear, well-lit face photo</div>
                </div>
                """, unsafe_allow_html=True)

        # ── RESULTS ─────────────────────────────────────────────────────
        if recs:
            st.success(f"Here are {len(recs)} songs based on detected mood:")
            for i, r in enumerate(recs):
                song_card(r, username, i)

        st.markdown("---")
        st.caption("💡 Tip: For best face detection results, use good lighting and ensure your face is clearly visible.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 – PLAYLISTS
    # ══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Your Playlists")
        with st.expander("➕ Create New Playlist"):
            new_pl = st.text_input("Playlist Name", placeholder="My Chill Vibes…")
            if st.button("Create"):
                if new_pl.strip():
                    save_playlist(username, new_pl.strip(), [])
                    st.success(f"✅ Playlist '{new_pl}' created!")
                    st.rerun()
                else:
                    st.warning("Enter a name.")

        playlists = get_user_playlists(username)
        if not playlists:
            st.info("No playlists yet. Create one above, then save songs from Discover!")
        else:
            for pl_name, songs in playlists.items():
                with st.expander(f"📋 {pl_name}  ({len(songs)} songs)"):
                    if songs:
                        for s in songs:
                            q = urllib.parse.quote_plus(f"{s['song']} {s['artist']}")
                            st.markdown(f"""
                            <div class='pl-item'>
                                <div>
                                    <div class='pl-song'>🎵 {s['song']}</div>
                                    <div class='pl-artist'>{s['artist']} · {s.get('emotion','—')}</div>
                                </div>
                                <div>
                                    <a href='https://open.spotify.com/search/{q}' target='_blank'
                                       style='color:#1DB954;font-size:.8rem;text-decoration:none;margin-right:10px'>▶ Spotify</a>
                                    <a href='https://www.youtube.com/results?search_query={q}' target='_blank'
                                       style='color:#FF0000;font-size:.8rem;text-decoration:none'>▶ YT</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("Empty playlist. Add songs from Discover!")
                    if st.button(f"🗑️ Delete '{pl_name}'", key=f"del_{pl_name}"):
                        delete_playlist(username, pl_name)
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 – ANALYTICS
    # ══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Your Mood Analytics")
        history  = get_history(username)
        feedback = get_feedback(username)

        if not history:
            st.info("No history yet. Use Discover to build your mood profile!")
        else:
            sessions, saved, top_e, num_pl = compute_stats(username)
            m1, m2, m3, m4 = st.columns(4)
            for col, val, lbl in zip([m1,m2,m3,m4],
                [sessions, saved, num_pl, len(feedback)],
                ["Sessions","Saved Songs","Playlists","Ratings Given"]):
                col.markdown(f"<div class='metric-card'><div class='val'>{val}</div>"
                             f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🎭 Emotion Distribution")
            ec_df = pd.DataFrame(
                Counter(h["emotion"] for h in history).items(),
                columns=["Emotion","Count"]
            ).sort_values("Count", ascending=False)
            try:
                import plotly.express as px
                fig = px.bar(ec_df, x="Emotion", y="Count", color="Emotion",
                             color_discrete_map={k:v["color"] for k,v in EMOTION_MAP.items()},
                             template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)",showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(ec_df.set_index("Emotion"))

            st.markdown("#### 📡 Detection Method Breakdown")
            md_df = pd.DataFrame(Counter(h["input_method"] for h in history).items(),
                                 columns=["Method","Count"])
            try:
                import plotly.express as px
                fig_m = px.pie(md_df, values="Count", names="Method", template="plotly_white",
                               color_discrete_sequence=["#0077aa","#228855","#FFD700","#FF69B4"])
                fig_m.update_layout(paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)")
                st.plotly_chart(fig_m, use_container_width=True)
            except ImportError:
                st.bar_chart(md_df.set_index("Method"))

            st.markdown("#### 🕐 Recent Mood History")
            for h in reversed(history[-15:]):
                ts     = h["timestamp"][:16].replace("T"," ")
                meta   = EMOTION_MAP.get(h["emotion"],{"emoji":"🎵"})
                m_icon = {"text":"✍️","face":"📷","weather":"🌦️","upload":"📂"}.get(h["input_method"],"🎵")
                st.markdown(f"""
                <div class='history-item'>
                    <div class='history-time'>{ts} · {m_icon} {h['input_method'].title()}</div>
                    <div class='history-mood'>{meta['emoji']} {h['emotion']} — {h['mood']}</div>
                </div>
                """, unsafe_allow_html=True)

            if feedback:
                st.markdown("#### ⭐ Your Song Ratings")
                fb_df = pd.DataFrame(feedback)[["song","artist","rating","comment"]]
                st.dataframe(fb_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 – EXPLORE
    # ══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Explore Dataset & Find Similar Songs")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown(f"#### 📊 Dataset: {len(df):,} songs loaded")
            ec = df["emotion"].value_counts().reset_index()
            ec.columns = ["Emotion","Count"]
            try:
                import plotly.express as px
                fig2 = px.pie(ec, values="Count", names="Emotion", template="plotly_white",
                              color_discrete_sequence=["#FFD700","#FF69B4","#87CEEB","#6495ED","#8B5CF6"])
                fig2.update_layout(paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)")
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(ec.set_index("Emotion"))

        with col_e2:
            st.markdown("#### 🔍 Find Lyrically Similar Songs")
            st.caption("TF-IDF cosine similarity on lyrics")
            search_song = st.text_input("Enter a song title from the dataset")
            if st.button("🔍 Find Similar"):
                if search_song.strip():
                    with st.spinner("Calculating similarity…"):
                        sims = content_similar(search_song)
                    if sims:
                        for s in sims:
                            q = urllib.parse.quote_plus(f"{s['song']} {s['artist']}")
                            st.markdown(f"""
                            <div class='pl-item'>
                                <div>
                                    <div class='pl-song'>🎵 {s['song']}</div>
                                    <div class='pl-artist'>{s['artist']} · {s['similarity']}% lyric match · {s['genre']}</div>
                                </div>
                                <a href='https://open.spotify.com/search/{q}' target='_blank'
                                   style='color:#1DB954;font-size:.8rem;text-decoration:none'>▶ Play</a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Song not found in dataset.")

        st.markdown("---")
        st.markdown("#### 🎸 Genre Breakdown")
        gc = df["genre"].value_counts().reset_index()
        gc.columns = ["Genre","Count"]
        try:
            import plotly.express as px
            fig3 = px.bar(gc, x="Genre", y="Count", template="plotly_white",
                          color="Count", color_continuous_scale=["#b0c4de","#0077aa"])
            fig3.update_layout(paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)",showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            st.bar_chart(gc.set_index("Genre"))

        st.markdown("#### 📈 Sentiment Score Distribution")
        try:
            import plotly.express as px
            fig4 = px.histogram(df, x="sentiment_score", nbins=50, template="plotly_white",
                                color_discrete_sequence=["#0077aa"])
            fig4.update_layout(paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)")
            st.plotly_chart(fig4, use_container_width=True)
        except ImportError:
            st.bar_chart(df["sentiment_score"].value_counts())

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 – SETTINGS
    # ══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("Settings & About")

        st.markdown("#### 📥 Export Your Data")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export Mood History (CSV)"):
                hist = get_history(username)
                if hist:
                    st.download_button("⬇ Download History CSV",
                        pd.DataFrame(hist).to_csv(index=False), "mood_history.csv","text/csv")
                else: st.info("No history yet.")
        with c2:
            if st.button("Export Playlists (CSV)"):
                pl   = get_user_playlists(username)
                rows = [{"playlist":name,**s} for name,songs in pl.items() for s in songs]
                if rows:
                    st.download_button("⬇ Download Playlists CSV",
                        pd.DataFrame(rows).to_csv(index=False), "playlists.csv","text/csv")
                else: st.info("No playlists yet.")

        st.markdown("---")
        st.markdown("#### 🗑️ Clear My Data")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Clear All History", type="primary"):
                d = _load_json(HISTORY_FILE, {}); d[username] = []; _save_json(HISTORY_FILE, d)
                st.success("History cleared.")
        with c4:
            if st.button("Clear All Ratings"):
                d = _load_json(FEEDBACK_FILE, {}); d[username] = []; _save_json(FEEDBACK_FILE, d)
                st.success("Ratings cleared.")

        st.markdown("---")
        st.markdown("#### ℹ️ About MoodTunes v2.0")
        st.markdown("""
**Mood Detection (4 Methods):**
- ✍️ **Text** — VADER + TextBlob dual NLP sentiment
- 📷 **Face Camera** — DeepFace deep learning emotion recognition
- 🌦️ **Weather** — Live weather via Open-Meteo API
- 📂 **Upload Image** — Drag & drop any face photo for DeepFace analysis *(new)*

**All Features:**
🎵 Emotion-based recommendations · 🎸 Genre filter · 🔀 Artist diversity mode ·
📋 Playlist management · 🔍 Lyric similarity search (TF-IDF) ·
▶️ Spotify + YouTube links · 📊 Plotly analytics · 🕐 Session history ·
⭐ Song ratings · 📥 CSV export · 🔐 Secure SHA-256 auth

**Stack:** Python · Streamlit · VADER · TextBlob · DeepFace · Scikit-learn · Plotly · Open-Meteo
        """)
        st.caption("MoodTunes v2.0 · Major Project Edition")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    main_app(st.session_state["username"])