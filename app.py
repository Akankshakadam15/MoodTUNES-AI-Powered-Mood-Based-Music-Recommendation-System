#╔══════════════════════════════════════════════════════════════════════════╗
# ║        MOODTUNES – AI Powered Mood-Based Music Recommendation System    ║
# ║                    v5.5 FINAL – Fixed Face Detection                    ║
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

st.set_page_config(
    page_title="MoodTunes – AI Music Recommender",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="auto",
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
USER_FILE     = os.path.join(BASE_DIR, "users.csv")
PLAYLIST_FILE = os.path.join(BASE_DIR, "playlists.json")
HISTORY_FILE  = os.path.join(BASE_DIR, "history.json")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#fff;color:#1a1a2e;}
.stApp{background:#fff;}
h1,h2,h3{font-family:'Orbitron',monospace;color:#1a1a2e;}
[data-testid="stSidebar"]{background:#f0f4f8!important;border-right:1px solid #d0d8e4;}
.metric-card{background:#f7f9fc;border:1px solid #b0c4de;border-radius:12px;padding:18px 22px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.07);margin-bottom:12px;}
.metric-card .val{font-family:'Orbitron',monospace;font-size:2rem;color:#0077aa;}
.metric-card .lbl{font-size:.8rem;color:#555577;margin-top:4px;text-transform:uppercase;letter-spacing:.08em;}
.song-card{background:#f7f9fc;border:1px solid #b0c4de;border-radius:14px;padding:20px 24px;margin-bottom:16px;transition:all .25s;}
.song-card:hover{border-color:#0077aa;box-shadow:0 4px 16px rgba(0,119,170,.12);}
.song-title{font-family:'Orbitron',monospace;font-size:1.05rem;color:#0077aa;}
.song-artist{color:#228855;font-size:.9rem;margin-top:2px;}
.emotion-badge{display:inline-block;background:#e8f0fe;border:1px solid #0077aa55;border-radius:20px;padding:3px 12px;font-size:.75rem;color:#0077aa;margin-top:8px;}
.match-bar-bg{background:#e0e8f0;border-radius:4px;height:6px;margin-top:8px;}
.match-bar-fill{background:linear-gradient(90deg,#0077aa,#00bb77);border-radius:4px;height:6px;}
.mood-display{background:#f0f7ff;border:1px solid #0077aa44;border-radius:16px;padding:24px;text-align:center;margin-bottom:20px;}
.mood-emoji{font-size:3.5rem;}
.mood-label{font-family:'Orbitron',monospace;font-size:1.4rem;color:#0077aa;margin-top:8px;}
.mood-score{color:#228855;font-size:.9rem;margin-top:4px;}
.stButton>button{background:linear-gradient(90deg,#0077aa,#00aa77)!important;color:#fff!important;font-family:'Orbitron',monospace!important;font-weight:700!important;border:none!important;border-radius:8px!important;padding:10px 24px!important;transition:all .2s!important;}
.stButton>button:hover{opacity:.85;box-shadow:0 4px 14px rgba(0,119,170,.3)!important;}
.stTabs [data-baseweb="tab-list"]{background:#f0f4f8;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#555577!important;font-family:'Orbitron',monospace!important;font-size:.75rem;}
.stTabs [aria-selected="true"]{background:#fff!important;color:#0077aa!important;border-radius:7px;box-shadow:0 1px 4px rgba(0,0,0,.1);}
.stTextInput input,.stTextArea textarea{background:#fff!important;color:#1a1a2e!important;border:1px solid #b0c4de!important;border-radius:8px!important;}
.pl-item{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#f7f9fc;border:1px solid #d0d8e4;border-radius:8px;margin-bottom:8px;}
.pl-song{font-size:.9rem;color:#1a1a2e;}
.pl-artist{font-size:.78rem;color:#555577;}
.history-item{border-left:2px solid #0077aa44;padding:8px 16px;margin-bottom:10px;}
.history-time{font-size:.75rem;color:#888899;}
.history-mood{color:#228855;font-size:.85rem;font-weight:600;}
.rating-card{background:#f0f7ff;border:1px solid #0077aa33;border-radius:10px;padding:14px 18px;margin-bottom:10px;}
.rating-song{font-weight:600;color:#1a1a2e;font-size:.9rem;}
.rating-artist{color:#555577;font-size:.78rem;}
.rating-stars{color:#FFD700;font-size:1rem;margin-top:4px;}
.rating-comment{color:#228855;font-size:.8rem;font-style:italic;margin-top:4px;}
.rating-time{color:#aaaacc;font-size:.72rem;margin-top:2px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
    except Exception as e:
        st.error(f"[JSON load error] {path}: {e}")
    return default

def _save_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        st.error(f"[JSON save error] {path}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# USER AUTH
# ══════════════════════════════════════════════════════════════════════════
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
    df  = load_users()
    hsh = _hash(password)
    return not df[
        (df["username"]==username) &
        ((df["password"]==hsh) | (df["password"]==password))
    ].empty

def get_user_color(username):
    df  = load_users()
    row = df[df["username"]==username]
    return row.iloc[0]["avatar_color"] if not row.empty and "avatar_color" in row.columns else "#0077aa"


# ══════════════════════════════════════════════════════════════════════════
# PLAYLIST
# ══════════════════════════════════════════════════════════════════════════
def get_user_playlists(username):
    return _load_json(PLAYLIST_FILE, {}).get(username, {})

def save_playlist(username, pl_name, songs):
    data = _load_json(PLAYLIST_FILE, {})
    data.setdefault(username, {})[pl_name] = songs
    return _save_json(PLAYLIST_FILE, data)

def delete_playlist(username, pl_name):
    data = _load_json(PLAYLIST_FILE, {})
    if username in data and pl_name in data[username]:
        del data[username][pl_name]
        _save_json(PLAYLIST_FILE, data)

def add_song_to_playlist(username, pl_name, song_dict):
    data = _load_json(PLAYLIST_FILE, {})
    data.setdefault(username, {}).setdefault(pl_name, [])
    if song_dict["song"] in [s["song"] for s in data[username][pl_name]]:
        return False, "already_exists"
    data[username][pl_name].append(song_dict)
    ok = _save_json(PLAYLIST_FILE, data)
    return (True, "ok") if ok else (False, "save_failed")


# ══════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
# FEEDBACK / RATINGS
# ══════════════════════════════════════════════════════════════════════════
def save_feedback(username, song, artist, rating, comment=""):
    data = _load_json(FEEDBACK_FILE, {})
    data.setdefault(username, [])
    for existing in data[username]:
        if existing["song"] == str(song).strip() and existing["artist"] == str(artist).strip():
            existing.update({
                "rating":    int(rating),
                "comment":   str(comment).strip(),
                "timestamp": datetime.datetime.now().isoformat(),
            })
            return _save_json(FEEDBACK_FILE, data)
    data[username].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "song":    str(song).strip(),
        "artist":  str(artist).strip(),
        "rating":  int(rating),
        "comment": str(comment).strip(),
    })
    return _save_json(FEEDBACK_FILE, data)

def get_feedback(username):
    return _load_json(FEEDBACK_FILE, {}).get(username, [])

def get_song_feedback(username, song, artist):
    for fb in get_feedback(username):
        if fb["song"] == str(song).strip() and fb["artist"] == str(artist).strip():
            return fb
    return None


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_data():
    csv_path = os.path.join(BASE_DIR, "spotify_millsongdata.csv")
    df = pd.read_csv(csv_path)
    if len(df) > 8000:
        df = df.sample(8000, random_state=42).reset_index(drop=True)

    def preprocess(text):
        return re.sub(r"[^a-zA-Z\s]", "", str(text)).lower().strip()

    df["cleaned_text"]    = df["text"].apply(preprocess)
    _v = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["cleaned_text"].apply(lambda t: _v.polarity_scores(t)["compound"])
    df["textblob_score"]  = df["cleaned_text"].apply(lambda t: TextBlob(t).sentiment.polarity)

    def classify(s):
        if s >  0.5: return "Happy / Energetic"
        if s >  0.1: return "Romantic / Positive"
        if s > -0.1: return "Neutral / Calm"
        if s > -0.5: return "Sad"
        return "Deep Sad"

    df["emotion"] = df["sentiment_score"].apply(classify)

    if "genre" not in df.columns:
        genres = ["Pop","Rock","R&B","Hip-Hop","Classical","Jazz","Electronic","Country","Folk","Latin"]
        df["genre"] = np.random.default_rng(42).choice(genres, size=len(df))

    tfidf        = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"].fillna(""))
    cosine_sim   = cosine_similarity(tfidf_matrix)
    return df, tfidf, tfidf_matrix, cosine_sim

df, tfidf, tfidf_matrix, cosine_sim = load_data()
analyzer = SentimentIntensityAnalyzer()

EMOTION_MAP = {
    "Happy / Energetic":   {"emoji":"😄","color":"#FFD700"},
    "Romantic / Positive": {"emoji":"💖","color":"#FF69B4"},
    "Neutral / Calm":      {"emoji":"😌","color":"#87CEEB"},
    "Sad":                 {"emoji":"😢","color":"#6495ED"},
    "Deep Sad":            {"emoji":"💔","color":"#8B5CF6"},
}


# ══════════════════════════════════════════════════════════════════════════
# KEYWORD + VADER TEXT MOOD DETECTION
# ══════════════════════════════════════════════════════════════════════════
EMOTION_KEYWORDS = {
    "Happy / Energetic": [
        "happy","joy","joyful","excited","great","wonderful","amazing",
        "fantastic","awesome","cheerful","delighted","glad","ecstatic","blessed",
        "energetic","pumped","thrilled","elated","euphoric","good","enjoying","fun",
        "laugh","laughing","smile","smiling","positive","celebrate","celebration",
    ],
    "Sad": [
        "sad","unhappy","cry","crying","tears","depressed","depression","heartbroken",
        "lonely","alone","miss","missing","lost","grief","grieve","hopeless","pain",
        "hurt","sorrowful","miserable","gloomy","down","low","broken","devastated",
        "nervous","anxious","anxiety","worried","worry","scared","fear","afraid",
        "stressed","stress","frustrated","upset","disappoint","disappointed",
        "overwhelmed","exhausted","tired","drained","numb","empty","bored",
    ],
    "Deep Sad": [
        "suicidal","worthless","give up","giving up","no point","end it",
        "darkness","despair","despairing","angry","anger","furious","rage",
        "mad","hate","hatred","violent","irritated","annoyed","disgusted",
    ],
    "Neutral / Calm": [
        "calm","peaceful","relaxed","relax","chill","okay","fine","ok",
        "normal","neutral","meh","alright","average","nothing special",
    ],
    "Romantic / Positive": [
        "romantic","romance","in love","crush","attracted","beautiful",
        "caring","tender","affection","passionate","intimate","lovely","adore",
        "soulmate","partner","darling","sweetheart","kiss","hug",
    ],
}

def detect_mood_from_keywords(text):
    text_lower = text.lower()
    scores = {e: 0 for e in EMOTION_KEYWORDS}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def detect_user_mood_from_text(text):
    keyword_emotion = detect_mood_from_keywords(text)
    vader_compound  = analyzer.polarity_scores(text)["compound"]
    tb_score        = TextBlob(text).sentiment.polarity
    combined        = (vader_compound * 0.7) + (tb_score * 0.3)

    if keyword_emotion:
        score_map = {
            "Happy / Energetic":    0.8,
            "Romantic / Positive":  0.5,
            "Neutral / Calm":       0.0,
            "Sad":                 -0.5,
            "Deep Sad":            -0.8,
        }
        feeling_map = {
            "Happy / Energetic":   "Happy",
            "Romantic / Positive": "Romantic",
            "Neutral / Calm":      "Calm",
            "Sad":                 "Sad",
            "Deep Sad":            "Deep Sad",
        }
        return score_map[keyword_emotion], feeling_map[keyword_emotion], "Keyword+VADER", keyword_emotion

    if combined >  0.5:  return combined, "Happy",    "VADER", "Happy / Energetic"
    if combined >  0.15: return combined, "Positive", "VADER", "Romantic / Positive"
    if combined > -0.15: return combined, "Neutral",  "VADER", "Neutral / Calm"
    if combined > -0.5:  return combined, "Sad",      "VADER", "Sad"
    return combined, "Very Sad", "VADER", "Deep Sad"


# ══════════════════════════════════════════════════════════════════════════
# FACE EMOTION DETECTION — v5.5 PIXEL-RATIO METHOD
#
# ROOT CAUSE OF BUG: Haar smile cascade is notoriously unreliable.
# minNeighbors=22 was TOO HIGH  → missed real smiles → wrongly detected "Sad"
# minNeighbors=8  was TOO LOW   → false positives on neutral/sad faces
# There is NO single threshold that works reliably for all faces.
#
# FIX: Replace smile cascade entirely with pixel-ratio analysis:
#   1. Detect face with Haar (still reliable for face bounding box)
#   2. Isolate mouth ROI (bottom 30% of face)
#   3. Apply adaptive threshold to find bright/white pixels (teeth)
#   4. Calculate "teeth ratio" = bright pixels / total mouth pixels
#   5. Happy  → teeth_ratio > 0.08  (teeth clearly visible)
#   6. Also use eye-openness + face brightness as tiebreaker
# ══════════════════════════════════════════════════════════════════════════
def map_face_emotion_to_label(face_emotion):
    e = str(face_emotion).lower()
    if e == "happy":   return  0.8, "Happy",   "Energetic", "Happy / Energetic"
    if e == "sad":     return -0.5, "Sad",     "Calm",      "Sad"
    if e == "neutral": return  0.0, "Neutral", "Calm",      "Neutral / Calm"
    if e == "calm":    return  0.1, "Calm",    "Calm",      "Neutral / Calm"
    return 0.0, "Neutral", "Calm", "Neutral / Calm"


def _get_teeth_ratio(mouth_roi):
    """
    Returns ratio of bright (teeth) pixels in the mouth region.
    Uses Otsu thresholding — adaptive, works regardless of lighting.
    Higher ratio = more teeth visible = smiling / happy.
    """
    import cv2
    if mouth_roi is None or mouth_roi.size == 0:
        return 0.0
    # Otsu threshold to separate bright teeth from darker lips/skin
    _, thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_pixels = float(np.sum(thresh == 255))
    total_pixels  = float(thresh.size)
    return bright_pixels / total_pixels if total_pixels > 0 else 0.0


def _count_open_eyes(face_roi, fh, fw):
    """Counts open eyes in the upper half of the face ROI."""
    import cv2
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    upper_face  = face_roi[:int(fh * 0.55), :]
    eyes        = eye_cascade.detectMultiScale(
        upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
    )
    return len(eyes)


def detect_face_emotion_from_image(pil_image):
    """
    v5.5 — Pixel-ratio smile detection.
    No smile cascade. Uses Otsu-thresholded teeth-pixel ratio instead.
    Thresholds tuned for both webcam and uploaded photos.
    """
    try:
        import cv2

        # ── Resize to max 900px for speed ────────────────────────────────
        max_dim = 900
        w, h = pil_image.size
        if max(w, h) > max_dim:
            scale     = max_dim / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)))
            w, h      = pil_image.size

        img  = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE for contrast normalisation (helps with dark/dim photos)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Try multiple scaleFactor / minNeighbors to find a face
        faces = []
        for sf in [1.05, 1.1, 1.15, 1.2]:
            for mn in [3, 4, 5]:
                detected = face_cascade.detectMultiScale(
                    gray_eq, scaleFactor=sf, minNeighbors=mn, minSize=(60, 60)
                )
                if len(detected) > 0:
                    faces = detected
                    break
            if len(faces) > 0:
                break

        # ── PATH A: Face found ───────────────────────────────────────────
        if len(faces) > 0:
            # Pick largest face
            x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_roi      = gray_eq[y:y + fh, x:x + fw]

            # — Mouth ROI: bottom 30% of face, middle 60% width —
            mouth_y1  = int(fh * 0.65)
            mouth_y2  = int(fh * 0.92)
            mouth_x1  = int(fw * 0.20)
            mouth_x2  = int(fw * 0.80)
            mouth_roi = face_roi[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            teeth_ratio     = _get_teeth_ratio(mouth_roi)
            n_open_eyes     = _count_open_eyes(face_roi, fh, fw)
            face_brightness = float(np.mean(face_roi))
            face_variance   = float(np.var(face_roi))

            # ── Decision logic ───────────────────────────────────────────
            # teeth_ratio > 0.08 → teeth clearly showing → Happy
            # Lowered from 0.15 to 0.08 to catch natural smiles
            # (even partial tooth-show registers ~0.08–0.15 with Otsu)
            if teeth_ratio > 0.08:
                return "happy"

            # Bright, open-eyed, expressive face with some tooth hint → Happy
            if teeth_ratio > 0.05 and n_open_eyes >= 2 and face_brightness > 130:
                return "happy"

            # Sad indicators: low variance (flat expression), dark, eyes closed/droopy
            sad_score = 0
            if face_variance   < 1500:   sad_score += 1   # flat, unexpressive face
            if n_open_eyes     < 2:      sad_score += 1   # eyes closed / droopy
            if face_brightness < 95:     sad_score += 1   # dark/underexposed face
            if teeth_ratio     < 0.01:   sad_score += 1   # no teeth at all (closed mouth)

            if sad_score >= 3:
                return "sad"
            elif sad_score >= 2 and face_brightness < 110:
                return "sad"
            elif n_open_eyes >= 2 and face_brightness > 120:
                return "neutral"
            else:
                return "calm"

        # ── PATH B: No face detected → full-image brightness fallback ────
        # Useful for partial/profile/far-away photos
        # Analyse the central vertical strip for teeth brightness
        mouth_y1   = int(h * 0.40)
        mouth_y2   = int(h * 0.72)
        mouth_x1   = int(w * 0.30)
        mouth_x2   = int(w * 0.70)
        mouth_area = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

        if mouth_area.size == 0:
            return "neutral"

        # Use Otsu on the central strip
        teeth_ratio_b = _get_teeth_ratio(mouth_area)
        overall_mean  = float(np.mean(gray))
        overall_var   = float(np.var(gray))

        if teeth_ratio_b > 0.10:           # clear teeth → happy
            return "happy"
        elif teeth_ratio_b > 0.05 and overall_mean > 120:
            return "happy"
        elif overall_mean < 85 or overall_var < 400:
            return "sad"
        else:
            return "neutral"

    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# WEATHER MOOD
# ══════════════════════════════════════════════════════════════════════════
def weather_to_mood(weather_desc):
    w = weather_desc.lower()
    if any(k in w for k in ["clear","sun"]):              return  0.7, "Happy / Energetic"
    if any(k in w for k in ["cloud","overcast"]):         return  0.0, "Neutral / Calm"
    if any(k in w for k in ["rain","drizzle","shower"]):  return -0.4, "Sad"
    if any(k in w for k in ["storm","thunder","snow"]):   return -0.7, "Deep Sad"
    if any(k in w for k in ["fog","mist","haze"]):        return -0.1, "Neutral / Calm"
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
        if not geo.get("results"):
            return None, None
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        wx  = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true&forecast_days=1", timeout=5
        ).json()
        code = wx.get("current_weather",{}).get("weathercode",0)
        temp = wx.get("current_weather",{}).get("temperature","?")
        return _wmo_code(code), temp
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════
def recommend_by_emotion_label(emotion_label, input_score, user_text,
                                recommend_type, user_feeling, top_n=10,
                                genre_filter=None, diversity=False):
    pool = df[df["emotion"] == emotion_label].copy()
    if pool.empty:
        pool = df.copy()
        pool["sentiment_diff"] = abs(pool["sentiment_score"] - input_score)
        pool = pool.sort_values("sentiment_diff").head(top_n * 2)

    if genre_filter and genre_filter != "All":
        filtered = pool[pool["genre"] == genre_filter]
        if not filtered.empty:
            pool = filtered

    pool["sentiment_diff"] = abs(pool["sentiment_score"] - input_score)
    pool = pool.sort_values("sentiment_diff")

    if diversity:
        seen, rows = set(), []
        for _, row in pool.iterrows():
            if row["artist"] not in seen:
                rows.append(row)
                seen.add(row["artist"])
            if len(rows) >= top_n:
                break
        pool = pd.DataFrame(rows)
    else:
        pool = pool.head(top_n)

    results = []
    for _, row in pool.iterrows():
        match_pct = min(100, max(0, round((1 - abs(row["sentiment_score"] - input_score)/2)*100)))
        results.append({
            "song":    row["song"],
            "artist":  row["artist"],
            "emotion": row["emotion"],
            "genre":   row.get("genre","—"),
            "match":   match_pct,
        })
    return results, None

def get_recommendations(user_text, camera_image, prefer_camera=False,
                        top_n=10, genre_filter=None, diversity=False):
    if user_text and user_text.strip() and not prefer_camera:
        score, feeling, rtype, emotion = detect_user_mood_from_text(user_text)
        return recommend_by_emotion_label(emotion, score, user_text, rtype, feeling,
                                          top_n=top_n, genre_filter=genre_filter, diversity=diversity)
    if camera_image is not None:
        try:
            pil = Image.open(io.BytesIO(camera_image.getvalue()))
        except Exception:
            pil = Image.open(camera_image)
        dominant = detect_face_emotion_from_image(pil)
        if dominant is None:
            return None, "Could not detect face. Please ensure good lighting and a clear face photo."
        score, feeling, rtype, emotion = map_face_emotion_to_label(dominant)
        return recommend_by_emotion_label(emotion, score, f"Face:{dominant}", rtype, feeling,
                                          top_n=top_n, genre_filter=genre_filter, diversity=diversity)
    return None, "Please type how you feel or capture a face."

def content_similar(song_name, top_n=6):
    idx_list = df.index[df["song"].str.lower() == song_name.lower()].tolist()
    if not idx_list:
        return []
    idx    = idx_list[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [
        {"song":df.iloc[i]["song"],"artist":df.iloc[i]["artist"],
         "emotion":df.iloc[i]["emotion"],"genre":df.iloc[i].get("genre","—"),
         "similarity":round(s*100)}
        for i,s in scores
    ]

def compute_stats(username):
    history   = get_history(username)
    playlists = get_user_playlists(username)
    top_e     = Counter(h["emotion"] for h in history).most_common(1)
    return (len(history), sum(len(v) for v in playlists.values()),
            top_e[0][0] if top_e else "—", len(playlists))


# ══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════
def mood_card(emotion_label, score, label, method):
    meta = EMOTION_MAP.get(emotion_label, {"emoji":"🎵","color":"#0077aa"})
    st.markdown(f"""
    <div class='mood-display'>
        <div class='mood-emoji'>{meta['emoji']}</div>
        <div class='mood-label'>{emotion_label}</div>
        <div class='mood-score'>Detected: <b>{label}</b> &nbsp;·&nbsp; Score: {score:.2f} &nbsp;·&nbsp; Via: {method}</div>
    </div>""", unsafe_allow_html=True)


def song_card(r, username, idx):
    meta  = EMOTION_MAP.get(r["emotion"], {"emoji":"🎵"})
    match = r.get("match", 80)
    q     = urllib.parse.quote_plus(f"{r['song']} {r['artist']}")
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
            <a href='https://open.spotify.com/search/{q}' target='_blank'
               style='background:#1DB954;color:#000;border-radius:6px;padding:5px 12px;
                      font-size:.78rem;text-decoration:none;margin-right:8px;font-weight:600'>▶ Spotify</a>
            <a href='https://www.youtube.com/results?search_query={q}' target='_blank'
               style='background:#FF0000;color:#fff;border-radius:6px;padding:5px 12px;
                      font-size:.78rem;text-decoration:none;font-weight:600'>▶ YouTube</a>
        </div>
    </div>""", unsafe_allow_html=True)

    playlists = get_user_playlists(username)
    pl_names  = list(playlists.keys())
    if pl_names:
        col_a, col_b = st.columns([3,1])
        with col_a:
            chosen = st.selectbox("Playlist", pl_names, key=f"pls_{idx}", label_visibility="collapsed")
        with col_b:
            if st.button("➕ Add", key=f"add_{idx}"):
                song_dict = {"song":r["song"],"artist":r["artist"],
                             "emotion":r["emotion"],"genre":r.get("genre","—")}
                success, reason = add_song_to_playlist(username, chosen, song_dict)
                if success:
                    st.session_state[f"add_msg_{idx}"] = ("success",f"✅ Added **{r['song']}** to **{chosen}**!")
                elif reason == "already_exists":
                    st.session_state[f"add_msg_{idx}"] = ("warning","⚠️ Already in that playlist.")
                else:
                    st.session_state[f"add_msg_{idx}"] = ("error","❌ Save failed.")
        if f"add_msg_{idx}" in st.session_state:
            kind, msg = st.session_state[f"add_msg_{idx}"]
            if kind=="success": st.success(msg)
            elif kind=="warning": st.warning(msg)
            else: st.error(msg)
    else:
        st.caption("📋 No playlists yet — create one in the Playlists tab first.")

    safe  = re.sub(r"[^a-zA-Z0-9]","_",r["song"])[:40]
    s_key = f"rated_{username}_{safe}"
    if s_key not in st.session_state:
        existing_fb = get_song_feedback(username, r["song"], r["artist"])
        if existing_fb:
            st.session_state[s_key] = {"rating":existing_fb["rating"],
                                        "comment":existing_fb.get("comment","")}
    with st.expander(f"⭐ Rate · {r['song']}", expanded=False):
        if st.session_state.get(s_key):
            fb    = st.session_state[s_key]
            stars = "⭐"*fb["rating"]+"☆"*(5-fb["rating"])
            st.success(f"Rating saved! {stars}")
            if fb.get("comment"):
                st.caption(f"Your comment: *{fb['comment']}*")
            if st.button("✏️ Edit Rating", key=f"edit_{idx}"):
                del st.session_state[s_key]
                st.rerun()
        else:
            rating  = st.slider("Your rating",1,5,3,key=f"rating_{idx}",format="%d ⭐")
            comment = st.text_input("Comment (optional)",placeholder="e.g. Love this track!",key=f"comment_{idx}")
            if st.button("💾 Save Rating",key=f"save_{idx}"):
                ok = save_feedback(username,r["song"],r["artist"],rating,comment)
                if ok:
                    st.session_state[s_key] = {"rating":rating,"comment":comment}
                    st.rerun()
                else:
                    st.error("❌ Could not save rating.")


# ══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div style='text-align:center;padding:48px 0 24px'>
        <div style='font-family:Orbitron,monospace;font-size:3rem;color:#0077aa;letter-spacing:.1em'>🎧 MOODTUNES</div>
        <div style='color:#555577;margin-top:10px;font-size:1.05rem'>AI-Powered Emotion Music Recommender</div>
    </div>""", unsafe_allow_html=True)
    _, c, _ = st.columns([1,2,1])
    with c:
        t1, t2 = st.tabs(["🔑 Login","🆕 Sign Up"])
        with t1:
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Login →", use_container_width=True):
                if verify_user(u, p):
                    st.session_state.update({"logged_in":True,"username":u})
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        with t2:
            st.markdown("<br>", unsafe_allow_html=True)
            nu  = st.text_input("Choose Username", key="su_u")
            np_ = st.text_input("Choose Password", type="password", key="su_p")
            if st.button("Create Account →", use_container_width=True):
                ok, msg = save_user(nu, np_)
                (st.success if ok else st.error)(msg)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
def render_sidebar(username):
    color = get_user_color(username)
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:16px 0 20px'>
        <div style='width:60px;height:60px;border-radius:50%;background:{color};margin:0 auto 10px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:Orbitron,monospace;font-size:1.5rem;color:#fff;font-weight:900'>
            {username[0].upper()}</div>
        <div style='font-family:Orbitron,monospace;color:{color};font-size:.9rem'>{username}</div>
    </div>""", unsafe_allow_html=True)
    sessions, saved, top_e, num_pl = compute_stats(username)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class='metric-card'><div class='val'>{sessions}</div><div class='lbl'>Sessions</div></div>
    <div class='metric-card'><div class='val'>{saved}</div><div class='lbl'>Saved Songs</div></div>
    <div class='metric-card'><div class='val'>{num_pl}</div><div class='lbl'>Playlists</div></div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"**Favourite Mood:** {EMOTION_MAP.get(top_e,{}).get('emoji','🎵')} {top_e}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
    st.sidebar.caption("MoodTunes v5.5 · Pixel-Ratio Smile · OpenCV")


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
def main_app(username):
    render_sidebar(username)
    st.markdown("""
    <div style='margin-bottom:24px'>
        <span style='font-family:Orbitron,monospace;font-size:1.9rem;color:#0077aa'>🎧 MOODTUNES</span>
        <span style='color:#555577;font-size:.9rem;margin-left:14px'>AI Music Recommender · Personalised · Explainable</span>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["🎵 Discover","📋 Playlists","📊 Analytics","🔍 Explore","⚙️ Settings"])

    # ── TAB 1: DISCOVER ──────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Detect Your Mood & Get Songs")
        input_method = st.radio(
            "Detection Method",
            ["✍️ Type Your Mood","📷 Face Camera","🌦️ My City Weather","📂 Upload Image"],
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
                diversity = st.checkbox("Artist Diversity Mode", value=False)

        if "current_recs" not in st.session_state:
            st.session_state["current_recs"] = []

        # ── TEXT ─────────────────────────────────────────────────────────
        if "✍️" in input_method:
            c1, c2 = st.columns(2)
            with c1:
                user_input    = st.text_area("💬 How are you feeling?",
                    placeholder="e.g. I feel sad, I am happy, I feel nervous…", height=120)
                prefer_camera = st.checkbox("Prefer camera over typed mood?", value=False)
            with c2:
                st.markdown("📷 Capture face (optional)")
                camera_image = st.camera_input("Webcam capture", key="cam1")
            if st.button("🔮 Recommend Songs", use_container_width=True):
                with st.spinner("Analysing mood…"):
                    recs, error = get_recommendations(
                        user_input, camera_image,
                        prefer_camera=prefer_camera,
                        top_n=top_n, genre_filter=genre_filter, diversity=diversity,
                    )
                if error:
                    st.error(error)
                    st.session_state["current_recs"] = []
                elif recs:
                    st.session_state["current_recs"] = recs
                    if user_input.strip() and not prefer_camera:
                        sc, fl, method, em = detect_user_mood_from_text(user_input)
                        st.session_state["current_mood"] = (em, sc, fl, method)
                        log_history(username, fl, em, top_n, "text")
                    elif camera_image:
                        pil = Image.open(io.BytesIO(camera_image.getvalue()))
                        dom = detect_face_emotion_from_image(pil)
                        if dom:
                            sc, fl, _, em = map_face_emotion_to_label(dom)
                            st.session_state["current_mood"] = (em, sc, f"Face:{dom}", "OpenCV")
                            log_history(username, fl, em, top_n, "face")

        # ── FACE CAMERA ──────────────────────────────────────────────────
        elif "📷" in input_method:
            camera_image = st.camera_input("📷 Capture your face", key="cam2")
            if st.button("🔮 Analyse My Face", use_container_width=True):
                if camera_image:
                    with st.spinner("Detecting emotion…"):
                        pil = Image.open(io.BytesIO(camera_image.getvalue()))
                        dom = detect_face_emotion_from_image(pil)
                    if dom:
                        sc, fl, rt, em = map_face_emotion_to_label(dom)
                        st.session_state["current_mood"] = (em, sc, f"Face:{dom}", "OpenCV")
                        log_history(username, fl, em, top_n, "face")
                        recs, err = recommend_by_emotion_label(
                            em, sc, f"Face:{dom}", rt, fl,
                            top_n=top_n, genre_filter=genre_filter, diversity=diversity,
                        )
                        st.session_state["current_recs"] = recs or []
                        if err:
                            st.error(err)
                    else:
                        st.error("Could not detect a face. Try better lighting or move closer.")
                else:
                    st.warning("Please capture a photo first.")

        # ── WEATHER ──────────────────────────────────────────────────────
        elif "🌦️" in input_method:
            city = st.text_input("🏙️ Enter your city", placeholder="Mumbai, Delhi, London…")
            if st.button("🔮 Mood from Weather", use_container_width=True):
                if city.strip():
                    with st.spinner(f"Fetching weather for {city}…"):
                        desc, temp = get_weather(city)
                    if desc:
                        st.info(f"🌡️ **{city}**: {desc} · {temp}°C")
                        sc, em = weather_to_mood(desc)
                        st.session_state["current_mood"] = (em, sc, desc, "Weather (Open-Meteo)")
                        log_history(username, desc, em, top_n, "weather")
                        recs, err = recommend_by_emotion_label(
                            em, sc, f"Weather:{desc}", "Mood-based", desc,
                            top_n=top_n, genre_filter=genre_filter, diversity=diversity,
                        )
                        st.session_state["current_recs"] = recs or []
                        if err:
                            st.error(err)
                    else:
                        st.error("Could not fetch weather. Check city name.")
                else:
                    st.warning("Please enter a city name.")

        # ── UPLOAD IMAGE ─────────────────────────────────────────────────
        elif "📂" in input_method:
            uploaded_file = st.file_uploader(
                "Drop image here", type=["jpg","jpeg","png","webp"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                pil_image = Image.open(uploaded_file)
                c1, c2    = st.columns(2)
                with c1:
                    st.image(pil_image, caption="Uploaded Image", width=350)
                with c2:
                    st.markdown(
                        f"**File:** {uploaded_file.name}\n\n"
                        f"**Size:** {uploaded_file.size/1024:.1f} KB\n\n"
                        f"**Dim:** {pil_image.width}×{pil_image.height} px"
                    )
                if st.button("🔮 Detect Emotion from Image", use_container_width=True):
                    with st.spinner("Analysing your expression…"):
                        dom = detect_face_emotion_from_image(pil_image)
                    if dom:
                        sc, fl, rt, em = map_face_emotion_to_label(dom)
                        st.session_state["current_mood"] = (em, sc, f"Face:{dom}", "OpenCV (Upload)")
                        log_history(username, fl, em, top_n, "upload")
                        recs, err = recommend_by_emotion_label(
                            em, sc, f"Face:{dom}", rt, fl,
                            top_n=top_n, genre_filter=genre_filter, diversity=diversity,
                        )
                        st.session_state["current_recs"] = recs or []
                        if err:
                            st.error(err)
                    else:
                        st.error("Could not detect a face. Try a clearer, well-lit photo.")
            else:
                st.markdown("""
                <div style='border:2px dashed #b0c4de;border-radius:14px;padding:48px 24px;
                            text-align:center;background:#f7f9fc'>
                    <div style='font-size:2.5rem'>📂</div>
                    <div style='font-family:Orbitron,monospace;color:#0077aa;margin-top:12px'>
                        Drag &amp; Drop your image here</div>
                    <div style='font-size:.8rem;margin-top:8px'>Supports JPG, JPEG, PNG, WEBP</div>
                </div>""", unsafe_allow_html=True)

        # ── RENDER RECS ──────────────────────────────────────────────────
        recs = st.session_state.get("current_recs", [])
        if recs:
            if "current_mood" in st.session_state:
                em, sc, fl, method = st.session_state["current_mood"]
                mood_card(em, sc, fl, method)
            st.success(f"🎵 {len(recs)} songs matched your mood!")
            for i, r in enumerate(recs):
                song_card(r, username, i)
        st.markdown("---")
        st.caption("💡 Tip: Type exactly how you feel — 'I feel sad', 'I feel happy', 'I feel nervous'")

    # ── TAB 2: PLAYLISTS ─────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Your Playlists")
        with st.expander("➕ Create New Playlist"):
            new_pl = st.text_input("Playlist Name", placeholder="My Chill Vibes…")
            if st.button("Create Playlist"):
                if new_pl.strip():
                    ok = save_playlist(username, new_pl.strip(), [])
                    if ok:
                        st.success(f"✅ Playlist '{new_pl}' created!")
                        st.rerun()
                    else:
                        st.error("❌ Could not save.")
                else:
                    st.warning("Please enter a name.")
        playlists = get_user_playlists(username)
        if not playlists:
            st.info("No playlists yet. Create one above, then add songs from Discover!")
        else:
            for pl_name, songs in playlists.items():
                with st.expander(f"📋 {pl_name}  ({len(songs)} songs)"):
                    if songs:
                        for s in songs:
                            q = urllib.parse.quote_plus(f"{s['song']} {s['artist']}")
                            c1, c2, _ = st.columns([5,1,1])
                            with c1:
                                st.markdown(f"""
                                <div class='pl-item'>
                                    <div>
                                        <div class='pl-song'>🎵 {s['song']}</div>
                                        <div class='pl-artist'>{s['artist']} · {s.get('emotion','—')} · {s.get('genre','—')}</div>
                                    </div>
                                    <div>
                                        <a href='https://open.spotify.com/search/{q}' target='_blank'
                                           style='color:#1DB954;font-size:.8rem;text-decoration:none;margin-right:10px'>▶ Spotify</a>
                                        <a href='https://www.youtube.com/results?search_query={q}' target='_blank'
                                           style='color:#FF0000;font-size:.8rem;text-decoration:none'>▶ YT</a>
                                    </div>
                                </div>""", unsafe_allow_html=True)
                            with c2:
                                safe_song = re.sub(r"[^a-zA-Z0-9]","_",s["song"])[:20]
                                if st.button("🗑️",key=f"rm_{pl_name}_{safe_song}",help=f"Remove {s['song']}"):
                                    data = _load_json(PLAYLIST_FILE,{})
                                    if username in data and pl_name in data[username]:
                                        data[username][pl_name] = [
                                            x for x in data[username][pl_name] if x["song"]!=s["song"]
                                        ]
                                        _save_json(PLAYLIST_FILE, data)
                                        st.rerun()
                    else:
                        st.write("Empty playlist. Add songs from Discover!")
                    if st.button(f"🗑️ Delete '{pl_name}'",key=f"del_{pl_name}"):
                        delete_playlist(username, pl_name)
                        st.rerun()

    # ── TAB 3: ANALYTICS ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Your Mood Analytics")
        history  = get_history(username)
        feedback = get_feedback(username)

        if not history:
            st.info("No history yet. Use Discover to build your mood profile!")
        else:
            sessions, saved, top_e, num_pl = compute_stats(username)
            m1, m2, m3, m4 = st.columns(4)
            for col, val, lbl in zip(
                [m1,m2,m3,m4],
                [sessions, saved, num_pl, len(feedback)],
                ["Sessions","Saved Songs","Playlists","Ratings Given"],
            ):
                col.markdown(
                    f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            st.markdown("#### 🎭 Emotion Distribution")
            ec_df = (
                pd.DataFrame(Counter(h["emotion"] for h in history).items(), columns=["Emotion","Count"])
                .sort_values("Count", ascending=False)
            )
            try:
                import plotly.express as px
                fig = px.bar(ec_df, x="Emotion", y="Count", color="Emotion",
                             color_discrete_map={k:v["color"] for k,v in EMOTION_MAP.items()},
                             template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(ec_df.set_index("Emotion"))

            st.markdown("#### 📡 Detection Method Breakdown")
            md_df = pd.DataFrame(Counter(h["input_method"] for h in history).items(), columns=["Method","Count"])
            try:
                fig_m = px.pie(md_df, values="Count", names="Method", template="plotly_white",
                               color_discrete_sequence=["#0077aa","#228855","#FFD700","#FF69B4"])
                fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_m, use_container_width=True)
            except Exception:
                st.bar_chart(md_df.set_index("Method"))

            st.markdown("#### 🕐 Recent Mood History")
            for h in reversed(history[-15:]):
                ts   = h["timestamp"][:16].replace("T"," ")
                meta = EMOTION_MAP.get(h["emotion"],{"emoji":"🎵"})
                icon = {"text":"✍️","face":"📷","weather":"🌦️","upload":"📂"}.get(h["input_method"],"🎵")
                st.markdown(f"""<div class='history-item'>
                    <div class='history-time'>{ts} · {icon} {h['input_method'].title()}</div>
                    <div class='history-mood'>{meta['emoji']} {h['emotion']} — {h['mood']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⭐ Your Song Ratings")
        if not feedback:
            st.info("You haven't rated any songs yet.")
        else:
            avg_r = sum(f["rating"] for f in feedback)/len(feedback)
            top_r = max(feedback, key=lambda x: x["rating"])
            r1,r2,r3 = st.columns(3)
            r1.markdown(f"<div class='metric-card'><div class='val'>{len(feedback)}</div><div class='lbl'>Songs Rated</div></div>",unsafe_allow_html=True)
            r2.markdown(f"<div class='metric-card'><div class='val'>{avg_r:.1f} ⭐</div><div class='lbl'>Avg Rating</div></div>",unsafe_allow_html=True)
            r3.markdown(f"<div class='metric-card'><div class='val'>⭐{top_r['rating']}</div><div class='lbl'>Highest Rated</div></div>",unsafe_allow_html=True)
            try:
                import plotly.express as px
                rc  = Counter(f["rating"] for f in feedback)
                rd  = pd.DataFrame([(f"{k}⭐",v) for k,v in sorted(rc.items())],columns=["Stars","Count"])
                fig_r = px.bar(rd,x="Stars",y="Count",template="plotly_white",
                               color_discrete_sequence=["#FFD700"],title="Rating Distribution")
                fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
                st.plotly_chart(fig_r,use_container_width=True)
            except Exception:
                pass
            st.markdown("##### 📝 All Your Rated Songs")
            for fb in reversed(feedback):
                rating  = int(fb.get("rating",0))
                stars   = "⭐"*rating+"☆"*(5-rating)
                ts      = fb.get("timestamp","")[:16].replace("T"," ")
                comment = fb.get("comment","").strip()
                cmt_html = f"<div class='rating-comment'>💬 {comment}</div>" if comment else \
                           "<div class='rating-comment' style='color:#aaa'>No comment</div>"
                st.markdown(f"""
                <div class='rating-card'>
                    <div class='rating-song'>🎵 {fb['song']}</div>
                    <div class='rating-artist'>🎤 {fb['artist']}</div>
                    <div class='rating-stars'>{stars}</div>
                    {cmt_html}
                    <div class='rating-time'>🕐 {ts}</div>
                </div>""", unsafe_allow_html=True)
            with st.expander("📊 Table View"):
                fb_df = pd.DataFrame(feedback)[["song","artist","rating","comment","timestamp"]]
                fb_df["timestamp"] = fb_df["timestamp"].str[:16].str.replace("T"," ")
                st.dataframe(fb_df.sort_values("timestamp",ascending=False),use_container_width=True,hide_index=True)

    # ── TAB 4: EXPLORE ───────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Explore Dataset & Find Similar Songs")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### 📊 Dataset: {len(df):,} songs")
            ec = df["emotion"].value_counts().reset_index()
            ec.columns = ["Emotion","Count"]
            try:
                import plotly.express as px
                fig2 = px.pie(ec,values="Count",names="Emotion",template="plotly_white",
                              color_discrete_sequence=["#FFD700","#FF69B4","#87CEEB","#6495ED","#8B5CF6"])
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2,use_container_width=True)
            except ImportError:
                st.bar_chart(ec.set_index("Emotion"))
        with c2:
            st.markdown("#### 🔍 Lyric Similarity Search")
            search_song = st.text_input("Enter a song title from the dataset")
            if st.button("🔍 Find Similar"):
                if search_song.strip():
                    with st.spinner("Calculating…"):
                        sims = content_similar(search_song)
                    if sims:
                        for s in sims:
                            q = urllib.parse.quote_plus(f"{s['song']} {s['artist']}")
                            st.markdown(f"""
                            <div class='pl-item'>
                                <div>
                                    <div class='pl-song'>🎵 {s['song']}</div>
                                    <div class='pl-artist'>{s['artist']} · {s['similarity']}% · {s['genre']}</div>
                                </div>
                                <a href='https://open.spotify.com/search/{q}' target='_blank'
                                   style='color:#1DB954;font-size:.8rem;text-decoration:none'>▶ Play</a>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Song not found in dataset.")
        st.markdown("---")
        st.markdown("#### 🎸 Genre Breakdown")
        gc = df["genre"].value_counts().reset_index()
        gc.columns = ["Genre","Count"]
        try:
            import plotly.express as px
            fig3 = px.bar(gc,x="Genre",y="Count",template="plotly_white",
                          color="Count",color_continuous_scale=["#b0c4de","#0077aa"])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
            st.plotly_chart(fig3,use_container_width=True)
        except ImportError:
            st.bar_chart(gc.set_index("Genre"))
        st.markdown("#### 📈 Sentiment Distribution")
        try:
            import plotly.express as px
            fig4 = px.histogram(df,x="sentiment_score",nbins=50,template="plotly_white",
                                color_discrete_sequence=["#0077aa"])
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4,use_container_width=True)
        except ImportError:
            st.bar_chart(df["sentiment_score"].value_counts())

    # ── TAB 5: SETTINGS ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Settings & About")
        with st.expander("🔧 Debug Info"):
            st.code(f"""BASE_DIR      = {BASE_DIR}
PLAYLIST_FILE = exists={os.path.exists(PLAYLIST_FILE)}
FEEDBACK_FILE = exists={os.path.exists(FEEDBACK_FILE)}
HISTORY_FILE  = exists={os.path.exists(HISTORY_FILE)}""")
            if st.button("🔄 Show raw feedback.json"):
                fb = get_feedback(username)
                st.write(f"Total ratings: {len(fb)}")
                if fb: st.json(fb[-5:])
            if st.button("🔄 Show raw playlists.json"):
                pl = get_user_playlists(username)
                st.write(f"Playlists: {list(pl.keys())}")
                st.json({k:len(v) for k,v in pl.items()})

        st.markdown("#### 📥 Export Your Data")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export Mood History (CSV)"):
                hist = get_history(username)
                if hist:
                    st.download_button("⬇ Download History CSV",
                        pd.DataFrame(hist).to_csv(index=False),"mood_history.csv","text/csv")
                else:
                    st.info("No history yet.")
        with c2:
            if st.button("Export Playlists (CSV)"):
                pl   = get_user_playlists(username)
                rows = [{"playlist":name,**s} for name,songs in pl.items() for s in songs]
                if rows:
                    st.download_button("⬇ Download Playlists CSV",
                        pd.DataFrame(rows).to_csv(index=False),"playlists.csv","text/csv")
                else:
                    st.info("No playlists yet.")

        st.markdown("---")
        st.markdown("#### 🗑️ Clear My Data")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Clear All History", type="primary"):
                d = _load_json(HISTORY_FILE,{})
                d[username] = []
                _save_json(HISTORY_FILE, d)
                st.success("History cleared.")
        with c4:
            if st.button("Clear All Ratings"):
                d = _load_json(FEEDBACK_FILE,{})
                d[username] = []
                _save_json(FEEDBACK_FILE, d)
                for k in [k for k in st.session_state if k.startswith(f"rated_{username}_")]:
                    del st.session_state[k]
                st.success("Ratings cleared.")

        st.markdown("---")
        st.markdown("""
#### ℹ️ About MoodTunes v5.5 — FINAL

**✅ v5.5 Key Fix — Pixel-Ratio Smile Detection (No Smile Cascade)**

**Root cause of the bug in v5.4:**
The Haar smile cascade has no reliable `minNeighbors` value:
- `minNeighbors=22` → too strict → missed real smiles → wrong "Sad" on smiling faces ❌
- `minNeighbors=8`  → too loose → false positives on neutral/sad faces ❌
No single threshold works for all lighting, angles, and face types.

**v5.5 Solution — Otsu Pixel-Ratio Method:**
1. Detect face bounding box with Haar (still reliable for this)
2. Isolate mouth ROI: bottom 30% of face, middle 60% width
3. Apply Otsu adaptive threshold → separates bright teeth from dark lips/skin
4. `teeth_ratio` = bright pixels / total mouth pixels
5. `teeth_ratio > 0.08` → Happy 😄 (teeth clearly showing)
6. Eye-openness + face brightness used as secondary tiebreaker

**Text Detection — Keyword + VADER:**
- 😄 Happy: happy, joy, excited, great, awesome…
- 😢 Sad: sad, cry, nervous, anxious, worried, stressed, tired…
- 💔 Deep Sad: angry, furious, hopeless, despair…
- 😌 Calm: calm, relaxed, chill, okay, fine…
- 💖 Romantic: romantic, in love, crush, adore…

**Stack:** Python · Streamlit · Keyword+VADER · TextBlob · OpenCV · Scikit-learn · Plotly
        """)
        st.caption("MoodTunes v5.5 Final · Pixel-Ratio Smile · No TensorFlow · Python 3.14 ✅")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    main_app(st.session_state["username"])
