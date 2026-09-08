#!/usr/bin/env python3
"""
Stream Sentiment Dashboard
Analyzes sentiment of YouTube live chat replay and/or video comments.

Auth:      @segaamerica.com OTP via AWS SES -> HMAC-signed URL token (?t=)
Sentiment: Claude Haiku via AWS Bedrock (batched, per-message scoring)
           + Claude Sonnet via AWS Bedrock (AI theme summary)
Sources:   pytchat (live chat) / youtube-comment-downloader (comments)
           or a JSON file exported earlier
"""

import base64
import hashlib
import hmac
import io
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_DOMAIN = "@segaamerica.com"
OTP_EXPIRY     = 600        # 10 minutes
TOKEN_EXPIRY   = 86400      # 1 day
MAX_ATTEMPTS   = 5

SUMMARY_MODEL_ID   = "us.anthropic.claude-sonnet-4-6"          # AI Insights narrative summary
SENTIMENT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # per-message sentiment scoring

POS_THRESHOLD  = 0.05       # compound score >= this -> positive
NEG_THRESHOLD  = -0.05      # compound score <= this -> negative

SENTIMENT_BATCH_SIZE  = 40  # messages per Haiku scoring call
SENTIMENT_MAX_WORKERS = 6   # concurrent scoring calls

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stream Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #080808; }
    [data-testid="stSidebar"]  { background-color: #0c0c0c; border-right: 1px solid #181818; }
    [data-testid="stHeader"]   { background: transparent; }

    h1, h2, h3, h4 { color: #f0f0f0 !important; letter-spacing: 0.3px; }
    p, li, label, .stMarkdown { color: #c8c8c8; }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111111, #0a0a0a);
        border: 1px solid #202020;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetric"] label { color: #888 !important; }
    div[data-testid="stMetricValue"] { color: #f5f5f5 !important; }

    /* Buttons */
    .stButton > button {
        background-color: #d92b2b;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
    }
    .stButton > button:hover { background-color: #b81f1f; color: #fff; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #999; }
    .stTabs [aria-selected="true"] { color: #f0f0f0 !important; border-bottom-color: #d92b2b !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #202020; border-radius: 8px; }

    /* Chat/comment quote cards */
    .quote-card {
        background: #101010;
        border-left: 3px solid #303030;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .quote-pos { border-left-color: #2fbf71; }
    .quote-neg { border-left-color: #d92b2b; }
    .quote-meta { color: #777; font-size: 0.8rem; margin-bottom: 3px; }
    .quote-text { color: #e5e5e5; font-size: 0.92rem; }

    a { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Auth — OTP via AWS SES, HMAC-signed URL token
# ─────────────────────────────────────────────────────────────────────────────

def _sign(payload: str) -> str:
    key = st.secrets["COOKIE_SIGNING_KEY"].encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()

def make_token(email: str) -> str:
    expiry = int(time.time()) + TOKEN_EXPIRY
    payload = f"{email}|{expiry}"
    sig = _sign(payload)
    raw = f"{payload}|{sig}".encode()
    return base64.urlsafe_b64encode(raw).decode()

def verify_token(token: str):
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, expiry, sig = raw.split("|")
        if not hmac.compare_digest(_sign(f"{email}|{expiry}"), sig):
            return None
        if int(expiry) < time.time():
            return None
        return email
    except Exception:
        return None

def send_otp_email(email: str, otp: str):
    import boto3
    ses = boto3.client(
        "ses",
        region_name=st.secrets["AWS_SES_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )
    ses.send_email(
        Source=st.secrets["EMAIL_FROM"],
        Destination={"ToAddresses": [email]},
        Message={
            "Subject": {"Data": "Your Stream Sentiment Dashboard code"},
            "Body": {
                "Text": {
                    "Data": f"Your one-time verification code is: {otp}\n\n"
                            f"This code expires in {OTP_EXPIRY // 60} minutes."
                }
            },
        },
    )

def login_gate():
    """Blocks the rest of the app until a verified session/token exists."""
    query_token = st.query_params.get("t")
    if query_token:
        email = verify_token(query_token)
        if email:
            st.session_state["auth_email"] = email
            return

    if st.session_state.get("auth_email"):
        return

    st.markdown("## 📊 Stream Sentiment Dashboard")
    st.caption("Sign in with your Sega America email to continue.")

    step = st.session_state.get("auth_step", "email")

    if step == "email":
        with st.form("email_form"):
            email = st.text_input("Work email", placeholder="you@segaamerica.com")
            submitted = st.form_submit_button("Send code")
        if submitted:
            email = email.strip().lower()
            if not email.endswith(ALLOWED_DOMAIN):
                st.error(f"Please use an {ALLOWED_DOMAIN} address.")
            else:
                otp = f"{random.randint(0, 999999):06d}"
                try:
                    send_otp_email(email, otp)
                except Exception as e:
                    st.error(f"Couldn't send the code: {e}")
                    st.stop()
                st.session_state["pending_email"] = email
                st.session_state["pending_otp"] = otp
                st.session_state["otp_expires"] = time.time() + OTP_EXPIRY
                st.session_state["otp_attempts"] = 0
                st.session_state["auth_step"] = "otp"
                st.rerun()

    elif step == "otp":
        st.info(f"We sent a 6-digit code to **{st.session_state['pending_email']}**.")
        with st.form("otp_form"):
            code = st.text_input("Verification code", max_chars=6)
            c1, c2 = st.columns([1, 1])
            submitted = c1.form_submit_button("Verify")
            resend = c2.form_submit_button("Resend / use a different email")
        if resend:
            st.session_state["auth_step"] = "email"
            st.rerun()
        if submitted:
            if time.time() > st.session_state.get("otp_expires", 0):
                st.error("That code expired. Please request a new one.")
                st.session_state["auth_step"] = "email"
            elif st.session_state.get("otp_attempts", 0) >= MAX_ATTEMPTS:
                st.error("Too many attempts. Please request a new code.")
                st.session_state["auth_step"] = "email"
            elif code.strip() == st.session_state.get("pending_otp"):
                email = st.session_state["pending_email"]
                st.session_state["auth_email"] = email
                token = make_token(email)
                st.query_params["t"] = token
                st.rerun()
            else:
                st.session_state["otp_attempts"] = st.session_state.get("otp_attempts", 0) + 1
                st.error("Incorrect code. Please try again.")

    st.stop()

login_gate()

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment engine — Claude Haiku via Bedrock (batched + concurrent)
# ─────────────────────────────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

def clean_for_sentiment(text: str) -> str:
    if not text:
        return ""
    # Strip YouTube custom emote shortcodes like :_yt-member-1: and excess whitespace
    text = re.sub(r":[a-zA-Z0-9_\-]+:", " ", text)
    return text.strip()

DEFAULT_TOPIC_CATEGORIES = ["Games", "Characters", "Features", "Music", "Requests"]
OTHER_TOPIC = "Other"

def _normalize_topic_categories(categories) -> list:
    """Dedupe (order-preserving) and guarantee the 'Other' catch-all is present,
    so the model always has somewhere to put messages that don't fit."""
    seen = []
    for c in categories or DEFAULT_TOPIC_CATEGORIES:
        c = str(c).strip()
        if c and c not in seen:
            seen.append(c)
    if OTHER_TOPIC not in seen:
        seen.append(OTHER_TOPIC)
    return seen

def _classification_system_prompt(topic_categories: list) -> str:
    cats = " | ".join(f'"{c}"' for c in topic_categories)
    return (
        "You are a classification engine for YouTube live chat and video comments. "
        "For each item, do two things from the audience's point of view:\n"
        "1. Sentiment: account for slang, sarcasm, emojis, all-caps enthusiasm, and internet "
        "shorthand (e.g. 'W', 'L', 'mid', 'based', 'cooked', 'ratio', 'fr fr', 'no cap'). "
        "Off-topic, purely factual, or spam messages should be Neutral.\n"
        f"2. Topic: assign exactly one topic from this fixed list: {cats}. Use \"{OTHER_TOPIC}\" "
        "for greetings, generic hype/reactions, spam, or anything that doesn't clearly fit one "
        "of the other categories — don't force a fit.\n"
        "Respond with ONLY a JSON array, no prose, no markdown code fences: "
        '[{"id": <int>, "label": "Positive"|"Neutral"|"Negative", "score": <float from -1.0 to 1.0>, '
        '"topic": <one of the topic strings above>}, ...] '
        "— exactly one object per input item, reusing the exact same ids you were given."
    )

def _parse_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)

SCORE_BATCH_MAX_ATTEMPTS = 3
SCORE_BATCH_BACKOFF_BASE = 0.6  # seconds; doubles each retry (0.6s, 1.2s)

def _score_batch(batch_items, topic_categories=None):
    """batch_items: list of (idx, text). Returns dict idx -> (score, label, topic).
    Empty dict on failure (after retries)."""
    topic_categories = _normalize_topic_categories(topic_categories)
    payload = json.dumps([
        {"id": i, "text": clean_for_sentiment(t)[:500]} for i, t in batch_items
    ])
    client = get_bedrock_client()
    system = _classification_system_prompt(topic_categories)
    for attempt in range(SCORE_BATCH_MAX_ATTEMPTS):
        try:
            resp = client.messages.create(
                model=SENTIMENT_MODEL_ID,
                max_tokens=min(4000, len(batch_items) * 40 + 200),
                system=system,
                messages=[{"role": "user", "content": payload}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text")
            parsed = _parse_json_array(raw)
            out = {}
            for item in parsed:
                i = int(item["id"])
                label = item.get("label", "Neutral")
                if label not in ("Positive", "Neutral", "Negative"):
                    label = "Neutral"
                score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
                topic = item.get("topic", OTHER_TOPIC)
                if topic not in topic_categories:
                    topic = OTHER_TOPIC
                out[i] = (score, label, topic)
            return out
        except Exception:
            # Running 6 batches concurrently makes transient Bedrock throttling
            # more likely precisely because of the parallelism — an instant
            # retry with no delay just hits the same throttle window again.
            # Back off (0.6s, then 1.2s) before the next attempt so a real
            # rate limit has a chance to clear.
            if attempt < SCORE_BATCH_MAX_ATTEMPTS - 1:
                time.sleep(SCORE_BATCH_BACKOFF_BASE * (2 ** attempt))
            continue
    return {}

def score_sentiment(texts, status_cb=None, topic_categories=None):
    """Batched, concurrent sentiment + topic classification via Claude Haiku
    (Bedrock). Returns a list of (score, label, topic) aligned to `texts`. Any
    item whose batch fails outright (after retry) falls back to
    (0.0, "Neutral", "Other") rather than breaking the whole run."""
    topic_categories = _normalize_topic_categories(topic_categories)
    n = len(texts)
    results = [(0.0, "Neutral", OTHER_TOPIC)] * n
    if n == 0:
        return results

    indexed = [(i, t) for i, t in enumerate(texts) if t and str(t).strip()]
    if not indexed:
        return results

    batches = [indexed[i:i + SENTIMENT_BATCH_SIZE] for i in range(0, len(indexed), SENTIMENT_BATCH_SIZE)]
    done = 0
    failed_batches = 0

    with ThreadPoolExecutor(max_workers=SENTIMENT_MAX_WORKERS) as ex:
        futures = {ex.submit(_score_batch, b, topic_categories): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            batch_result = fut.result()
            if not batch_result:
                failed_batches += 1
            for i, _ in batch:
                if i in batch_result:
                    results[i] = batch_result[i]
            done += len(batch)
            if status_cb:
                status_cb(done, len(indexed))

    if failed_batches and status_cb:
        status_cb(len(indexed), len(indexed))
        st.session_state["_sentiment_failed_batches"] = failed_batches

    return results

class StreamingScorer:
    """Lets a fetch loop feed message text as it arrives and have it scored by
    Haiku *while more items are still being fetched*, instead of waiting for
    the whole download to finish before scoring anything.

    Batches fill up and get submitted to a background ThreadPoolExecutor as
    soon as they're full; those Bedrock calls run concurrently with whatever
    network waiting the fetch loop is still doing (both are I/O-bound, so
    Python releases the GIL for both), which is the main lever for cutting
    total wall-clock time on a big chat/comment pull."""

    def __init__(self, batch_size=SENTIMENT_BATCH_SIZE, max_workers=SENTIMENT_MAX_WORKERS,
                 status_cb=None, topic_categories=None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []       # list of (future, batch)
        self._pending = []       # (global_idx, text) not yet submitted
        self._n_fed = 0
        self._n_submitted = 0
        self._batch_size = batch_size
        self._status_cb = status_cb
        self._topic_categories = _normalize_topic_categories(topic_categories)

    def feed(self, text) -> int:
        """Call once per item as it arrives, in fetch order. Returns the item's
        index so callers can align results later (they don't need to — order
        is preserved automatically via finish())."""
        idx = self._n_fed
        self._n_fed += 1
        if text and str(text).strip():
            self._pending.append((idx, text))
            if len(self._pending) >= self._batch_size:
                self._flush_batch()
        return idx

    def _flush_batch(self):
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        self._n_submitted += len(batch)
        fut = self._executor.submit(_score_batch, batch, self._topic_categories)
        self._futures.append((fut, batch))

    def finish(self):
        """Flush any partial batch, wait for every in-flight scoring call, and
        return (scores, failed_batch_count). scores is a list of
        (score, label, topic) aligned to every text fed via feed(), in feed
        order."""
        self._flush_batch()
        results = {}
        failed = 0
        done = 0
        total = self._n_submitted
        for fut, batch in self._futures:
            batch_result = fut.result()
            if not batch_result:
                failed += 1
            for i, _ in batch:
                if i in batch_result:
                    results[i] = batch_result[i]
            done += len(batch)
            if self._status_cb:
                self._status_cb(done, max(total, 1))
        self._executor.shutdown(wait=True)

        out = [(0.0, "Neutral", OTHER_TOPIC)] * self._n_fed
        for i, val in results.items():
            out[i] = val
        return out, failed

def extract_emojis(text: str):
    if not text:
        return []
    return EMOJI_PATTERN.findall(text)

STOPWORDS = set("""
a an the and or but if is are was were be been being to of in on at for with
this that these those i you he she it we they my your his her its our their
me him her us them im ive youre its it's dont don't cant can't just so not no
yes lol lmao omg like really very much too so much can will would could should
have has had do does did as from up out about into over under again further
then once here there when where why how all any both each few more most other
some such only own same than too very s t just don should now
""".split())

def top_keywords(texts, n=25, min_len=3):
    counter = Counter()
    for t in texts:
        if not t:
            continue
        words = re.findall(r"[a-zA-Z']{%d,}" % min_len, t.lower())
        for w in words:
            if w not in STOPWORDS:
                counter[w] += 1
    return counter.most_common(n)

# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion — Live chat (pytchat) and JSON upload
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_chat(video_id: str, max_messages=None, max_seconds=None, progress_cb=None,
                     topchat_only=False, scorer=None):
    """Pull chat replay / live chat via pytchat.

    max_messages / max_seconds are optional safety caps — pass None (the
    default) to fetch everything. For an archived VOD replay, pytchat's
    is_alive() naturally goes False once it reaches the end, so an uncapped
    call still terminates on its own. A genuinely live (currently streaming)
    video has no defined end, so max_seconds is the only thing that will
    stop it if the person leaves it uncapped.

    topchat_only: pytchat native option that only returns YouTube's
    "top chat" (highlighted) messages instead of everything — meaningfully
    fewer network round-trips on very high-traffic streams, at the cost of
    completeness. Off by default since the default behavior is "analyze
    everything".

    scorer: an optional StreamingScorer. If given, each message's text is fed
    to it the instant it arrives, so Haiku scoring runs concurrently with the
    rest of the fetch instead of starting only after fetching finishes."""
    import pytchat

    # interruptable=False: pytchat registers a SIGINT handler by default, which
    # only works on the main interpreter thread. Streamlit runs script execution
    # on a worker thread, so this would otherwise raise
    # "ValueError: signal only works in main thread of the main interpreter".
    chat = pytchat.create(video_id=video_id, interruptable=False, topchat_only=topchat_only)
    items = []
    start = time.time()

    try:
        while chat.is_alive():
            if max_messages is not None and len(items) >= max_messages:
                break
            if max_seconds is not None and time.time() - start > max_seconds:
                break
            for c in chat.get().sync_items():
                items.append({
                    "datetime": c.datetime,
                    "elapsed_time": c.elapsedTime,
                    "author_name": c.author.name,
                    "author_channel_id": c.author.channelId,
                    "is_chat_owner": c.author.isChatOwner,
                    "is_chat_moderator": c.author.isChatModerator,
                    "is_chat_sponsor": c.author.isChatSponsor,
                    "message": c.message,
                    "message_id": c.id,
                })
                if scorer is not None:
                    scorer.feed(c.message)
                if progress_cb:
                    progress_cb(len(items))
                if max_messages is not None and len(items) >= max_messages:
                    break
    finally:
        try:
            chat.terminate()
        except Exception:
            pass

    return items

# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion — Twitch chat, live tail or VOD replay (chat-downloader)
# ─────────────────────────────────────────────────────────────────────────────

TWITCH_LIVE_HARD_TIMEOUT = 3600  # seconds; safety ceiling for an uncapped live tail

class TwitchChatError(Exception):
    """Raised with a clear, user-facing message for known Twitch failure modes
    (no chat replay available, invalid channel/VOD, etc.) instead of letting a
    library-internal exception surface as a raw traceback."""
    pass

def _normalize_twitch_message(msg: dict) -> dict:
    """Map a chat-downloader message dict (live IRC tags or VOD/GQL comment —
    the two paths expose author status slightly differently) onto the same
    chat item schema fetch_live_chat() produces, so chat_to_dataframe() works
    on Twitch data completely unchanged."""
    author = msg.get("author") or {}
    badges = author.get("badges") or []
    badge_names = {b.get("name") for b in badges if isinstance(b, dict) and b.get("name")}

    is_owner = "broadcaster" in badge_names
    is_moderator = bool(author.get("is_moderator")) or "moderator" in badge_names
    is_sponsor = (bool(author.get("is_subscriber")) or "subscriber" in badge_names
                  or "founder" in badge_names)

    ts_micros = msg.get("timestamp")
    if ts_micros:
        dt = datetime.fromtimestamp(ts_micros / 1_000_000, tz=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    return {
        "datetime": dt.isoformat(),
        "elapsed_time": msg.get("time_text"),
        "author_name": author.get("display_name") or author.get("name") or "Unknown",
        "author_channel_id": author.get("id"),
        "is_chat_owner": is_owner,
        "is_chat_moderator": is_moderator,
        "is_chat_sponsor": is_sponsor,
        "message": msg.get("message") or "",
        "message_id": msg.get("message_id") or f"{ts_micros}-{author.get('id')}",
    }

def fetch_twitch_chat(url: str, max_messages=None, max_seconds=None, progress_cb=None, scorer=None):
    """Fetch Twitch chat — live tail or VOD chat replay, auto-detected from the
    URL by chat-downloader — and normalize it into the same chat item schema
    used everywhere else in this app.

    A live channel URL (twitch.tv/channelname) tails chat in real time via
    Twitch's anonymous IRC protocol (no token or app registration needed for
    read-only access). A VOD URL (twitch.tv/videos/12345) or clip URL instead
    pages through the archived chat replay via the same internal GQL endpoint
    the twitch.tv website itself uses — same idea as pytchat's approach to
    YouTube. Both come back through the same generator, so this one function
    covers both cases; chat-downloader tells them apart from the URL alone.

    max_messages/max_seconds are optional caps (None = uncapped), mirroring
    fetch_live_chat(). The wall-clock deadline is only enforced when needed:
    a VOD/clip replay ends on its own once the archive is exhausted, so it
    stays fully uncapped unless the caller explicitly passes max_seconds — a
    hard ceiling (TWITCH_LIVE_HARD_TIMEOUT) only kicks in automatically for a
    genuinely live/upcoming channel, which otherwise has no natural end."""
    from chat_downloader import ChatDownloader
    from chat_downloader.errors import NoChatReplay, VideoUnavailable, UserNotFound

    downloader = ChatDownloader()
    try:
        # No `timeout=` passed here deliberately — that would make
        # chat-downloader apply one wall-clock deadline uniformly to both live
        # and VOD fetches. We want to know chat.status first (below) so a
        # finite VOD replay isn't cut off by a ceiling meant for endless live
        # streams; timeout is enforced by hand in the loop instead.
        chat = downloader.get_chat(url, max_messages=max_messages, message_receive_timeout=1.0)
    except NoChatReplay:
        raise TwitchChatError(
            "This Twitch VOD/clip doesn't have chat replay available (it may have expired, "
            "or the streamer disabled it)."
        )
    except UserNotFound:
        raise TwitchChatError("No Twitch channel found for that name — double check it's correct.")
    except VideoUnavailable:
        raise TwitchChatError("That Twitch VOD/clip ID doesn't exist or is unavailable.")

    is_live_status = getattr(chat, "status", None) in ("live", "upcoming")
    if max_seconds is not None:
        deadline = time.time() + max_seconds
    elif is_live_status:
        deadline = time.time() + TWITCH_LIVE_HARD_TIMEOUT
    else:
        deadline = None  # finite VOD/clip replay — let it run to completion

    items = []
    try:
        for msg in chat:
            if deadline is not None and time.time() >= deadline:
                break
            item = _normalize_twitch_message(msg)
            items.append(item)
            if scorer is not None:
                scorer.feed(item["message"])
            if progress_cb:
                progress_cb(len(items))
            if max_messages is not None and len(items) >= max_messages:
                break
    except (NoChatReplay, VideoUnavailable, UserNotFound):
        # can also surface mid-iteration depending on how the library resolves the URL
        if not items:
            raise TwitchChatError(
                "Twitch chat couldn't be retrieved for that channel/VOD — it may not exist, "
                "or chat replay isn't available for it."
            )
        # if we already collected some messages before the error, keep them rather than
        # discarding a partial (and possibly large) fetch over a late-surfacing error
    return items

def load_chat_json(raw_bytes) -> list:
    """Accepts the JSON structure produced by the user's download script (nested 'author')
    as well as a flat structure, and normalizes to the flat schema fetch_live_chat() produces."""
    data = json.loads(raw_bytes)
    if isinstance(data, dict):
        # tolerate {"items": [...]}-style wrappers
        for key in ("items", "chat", "messages", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    out = []
    for c in data:
        # No default here (None, not {}) so a flat-schema item that simply has no
        # "author" key at all falls through to the flat branch below, instead of
        # matching isinstance(author, dict) on an empty dict and silently losing
        # author_channel_id / owner / moderator / sponsor flags to it.
        author = c.get("author")
        if isinstance(author, dict) and author:
            out.append({
                "datetime": c.get("datetime"),
                "elapsed_time": c.get("elapsed_time") or c.get("elapsedTime"),
                "author_name": author.get("name") or c.get("author_name"),
                "author_channel_id": author.get("channel_id") or author.get("channelId"),
                "is_chat_owner": author.get("is_chat_owner", False),
                "is_chat_moderator": author.get("is_chat_moderator", False),
                "is_chat_sponsor": author.get("is_chat_sponsor", False),
                "message": c.get("message"),
                "message_id": c.get("message_id") or c.get("id"),
            })
        else:
            # already-flat schema
            out.append({
                "datetime": c.get("datetime"),
                "elapsed_time": c.get("elapsed_time"),
                "author_name": c.get("author_name") or author,
                "author_channel_id": c.get("author_channel_id"),
                "is_chat_owner": c.get("is_chat_owner", False),
                "is_chat_moderator": c.get("is_chat_moderator", False),
                "is_chat_sponsor": c.get("is_chat_sponsor", False),
                "message": c.get("message"),
                "message_id": c.get("message_id"),
            })
    return out

def chat_to_dataframe(items: list, status_cb=None, precomputed_scores=None,
                       topic_categories=None) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["source"] = "Chat"
    df["text"] = df["message"].fillna("")
    df["author"] = df["author_name"].fillna("Unknown")
    df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    scores = precomputed_scores if precomputed_scores is not None else \
        score_sentiment(df["text"].tolist(), status_cb=status_cb, topic_categories=topic_categories)
    df["sentiment_score"] = [s[0] for s in scores]
    df["sentiment_label"] = [s[1] for s in scores]
    df["topic"] = [s[2] if len(s) > 2 else OTHER_TOPIC for s in scores]
    df["badge"] = df.apply(
        lambda r: "Owner" if r.get("is_chat_owner") else
                  ("Moderator" if r.get("is_chat_moderator") else
                   ("Member" if r.get("is_chat_sponsor") else "Viewer")),
        axis=1,
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion — Comments (youtube-comment-downloader) and JSON upload
# ─────────────────────────────────────────────────────────────────────────────

def fetch_comments(video_id: str, max_comments=None, sort: str = "Popular", progress_cb=None, scorer=None):
    """Pull comments via youtube-comment-downloader. max_comments=None (default)
    fetches every comment the downloader can find — the generator ends on its
    own once it runs out, no artificial cap needed.

    scorer: an optional StreamingScorer — same idea as fetch_live_chat's, so
    Haiku scoring overlaps with the page-by-page comment fetching instead of
    starting only once every comment is in hand."""
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

    downloader = YoutubeCommentDownloader()
    sort_by = SORT_BY_POPULAR if sort == "Popular" else SORT_BY_RECENT
    url = f"https://www.youtube.com/watch?v={video_id}"

    items = []
    for c in downloader.get_comments_from_url(url, sort_by=sort_by):
        items.append(c)
        if scorer is not None:
            scorer.feed(c.get("text"))
        if progress_cb:
            progress_cb(len(items))
        if max_comments is not None and len(items) >= max_comments:
            break
    return items

def load_comments_json(raw_bytes) -> list:
    data = json.loads(raw_bytes)
    if isinstance(data, dict):
        for key in ("items", "comments", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if isinstance(data, list) and data and isinstance(data[0], str):
        # jsonl-as-list-of-strings edge case
        data = [json.loads(x) for x in data]
    return data

def comments_to_dataframe(items: list, status_cb=None, precomputed_scores=None,
                           topic_categories=None) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["source"] = "Comment"
    df["text"] = df.get("text", pd.Series(dtype=str)).fillna("")
    df["author"] = df.get("author", pd.Series(dtype=str)).fillna("Unknown")
    # youtube-comment-downloader gives unix-ish 'time_parsed' (epoch seconds) when available
    if "time_parsed" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_parsed"], unit="s", errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.NaT
    df["votes"] = pd.to_numeric(df.get("votes", 0), errors="coerce").fillna(0)
    df["reply_count"] = pd.to_numeric(df.get("replies", 0), errors="coerce").fillna(0)
    df["is_heart"] = df.get("heart", False)
    scores = precomputed_scores if precomputed_scores is not None else \
        score_sentiment(df["text"].tolist(), status_cb=status_cb, topic_categories=topic_categories)
    df["sentiment_score"] = [s[0] for s in scores]
    df["sentiment_label"] = [s[1] for s in scores]
    df["topic"] = [s[2] if len(s) > 2 else OTHER_TOPIC for s in scores]
    df["badge"] = df["is_heart"].apply(lambda h: "Creator ❤️" if h else "Viewer")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# AI theme summary — Claude via AWS Bedrock (map-reduce over chunks)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_bedrock_client():
    from anthropic import AnthropicBedrock
    return AnthropicBedrock(
        aws_region=st.secrets["AWS_BEDROCK_REGION"],
        aws_access_key=st.secrets["AWS_BEDROCK_ACCESS_KEY_ID"],
        aws_secret_key=st.secrets["AWS_BEDROCK_SECRET_ACCESS_KEY"],
    )

def _bedrock_call(system, user, max_tokens=1200):
    client = get_bedrock_client()
    resp = client.messages.create(
        model=SUMMARY_MODEL_ID,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")

TOPIC_DISCOVERY_SAMPLE_SIZE = 400

def discover_topic_categories(texts, min_categories: int = 4, max_categories: int = 6) -> list:
    """One Bedrock call: look at a sample of the actual message text and
    propose topic categories that describe what's really being discussed,
    instead of relying on a hand-typed list. Falls back to
    DEFAULT_TOPIC_CATEGORIES on any failure (empty input, bad response, API
    error) so a failed discovery call never blocks the rest of the run —
    scoring just proceeds with the generic defaults instead."""
    min_categories = max(1, min(min_categories, max_categories))
    non_empty = [str(t).strip() for t in texts if t and str(t).strip()]
    if not non_empty:
        return list(DEFAULT_TOPIC_CATEGORIES)

    if len(non_empty) > TOPIC_DISCOVERY_SAMPLE_SIZE:
        # Evenly spaced across the whole dataset rather than just the start,
        # so a sample from early in a long stream doesn't dominate.
        step = len(non_empty) / TOPIC_DISCOVERY_SAMPLE_SIZE
        sample = [non_empty[int(i * step)] for i in range(TOPIC_DISCOVERY_SAMPLE_SIZE)]
    else:
        sample = non_empty

    system = (
        "You are analyzing a sample of YouTube live chat / comment messages to propose a "
        "small set of topic categories that best describe what people are actually discussing "
        "— grounded in this specific sample's content, not a generic list. "
        f"Propose between {min_categories} and {max_categories} categories — use the low end if "
        "the content is genuinely narrow, and don't pad the list with weak/overlapping categories "
        "just to hit the upper bound. Keep each label short (1-3 words), title case, and "
        "non-overlapping. Don't include a catch-all/'Other' category — that's added automatically "
        "afterward for anything that doesn't fit. "
        "Respond with ONLY a JSON array of strings, no prose, no markdown fences."
    )
    user = "Sample messages:\n" + "\n".join(f"- {t[:200]}" for t in sample)
    try:
        raw = _bedrock_call(system, user, max_tokens=300)
        categories = _parse_json_array(raw)
        categories = [str(c).strip() for c in categories if str(c).strip()]
        if categories:
            return categories[:max_categories]
    except Exception:
        pass
    return list(DEFAULT_TOPIC_CATEGORIES)

def _sample_for_ai(df: pd.DataFrame, cap: int = 1200) -> pd.DataFrame:
    """Keep the AI pass fast/cheap: use everything under the cap, otherwise a
    stratified sample weighted toward higher-engagement / extreme-sentiment items."""
    if len(df) <= cap:
        return df
    extremes = df.reindex(df["sentiment_score"].abs().sort_values(ascending=False).index).head(cap // 2)
    rest = df.drop(extremes.index).sample(n=min(cap - len(extremes), len(df) - len(extremes)), random_state=42)
    return pd.concat([extremes, rest]).sort_index()

CHUNK_SIZE = 150  # messages per map-step

def generate_ai_summary(df: pd.DataFrame, video_id: str, label: str, status_cb=None):
    """Map-reduce: summarize chunks of chat/comment text, then synthesize a final
    narrative summary of themes, sentiment drivers, and notable quotes."""
    sample = _sample_for_ai(df)
    texts = [f"[{row.sentiment_label}] {row.author}: {row.text}" for row in sample.itertuples() if row.text]

    if not texts:
        return "Not enough text content to generate a summary."

    chunks = [texts[i:i + CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]
    chunk_notes = []

    system_map = (
        "You are analyzing a slice of YouTube audience reaction (chat and/or comments) for "
        "internal reporting. Be concise, concrete, and neutral. Note recurring topics, praise, "
        "complaints, running jokes/memes, and anything that looks like spam or brigading."
    )

    for i, chunk in enumerate(chunks):
        if status_cb:
            status_cb(f"Analyzing batch {i + 1} of {len(chunks)}...")
        block = "\n".join(chunk)[:12000]
        note = _bedrock_call(
            system_map,
            f"Video ID: {video_id}\nAudience feedback batch:\n{block}\n\n"
            f"In 4-6 bullet points, note the key recurring themes, notable praise/complaints, "
            f"and anything unusual in this batch.",
            max_tokens=500,
        )
        chunk_notes.append(note)

    if status_cb:
        status_cb("Synthesizing final summary...")

    system_reduce = (
        "You are writing the executive summary section of a YouTube audience sentiment report. "
        "Write in clear prose with short headers. Be specific and avoid generic filler."
    )
    synthesis = _bedrock_call(
        system_reduce,
        f"Here are analyst notes from {len(chunk_notes)} batches of {label} on video {video_id}:\n\n"
        + "\n\n".join(f"Batch {i+1}:\n{n}" for i, n in enumerate(chunk_notes))
        + "\n\nSynthesize this into a single report section with these headers: "
          "'Overall Sentiment', 'Key Themes', 'Notable Praise', 'Notable Complaints / Concerns', "
          "'Anything Unusual'. Keep it tight — a paragraph or short bullet list per header.",
        max_tokens=1400,
    )
    return synthesis

# ─────────────────────────────────────────────────────────────────────────────
# Chat with your data — Claude (Sonnet via Bedrock) + tool use over the dataframe
# ─────────────────────────────────────────────────────────────────────────────

CHAT_MAX_TOOL_ROUNDS = 4
CHAT_MAX_TOKENS = 1200

DATA_CHAT_TOOLS = [
    {
        "name": "search_messages",
        "description": (
            "Search and filter the FULL set of analyzed chat messages / comments (not just a "
            "sample) and return matching examples with their real text, author, sentiment, and "
            "source. Use this whenever you need actual quotes, examples, or exact counts of a "
            "filtered subset — never invent or paraphrase a quote without pulling it from here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Case-insensitive substring to search for in the message text. Omit to not filter by keyword."},
                "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"], "description": "Filter to only this sentiment label."},
                "source": {"type": "string", "enum": ["Chat", "Comment"], "description": "Filter to only live chat messages or only comments."},
                "topic": {"type": "string", "description": "Filter to only this topic category (one of the topic categories in play for this dataset)."},
                "author": {"type": "string", "description": "Case-insensitive substring match on author name."},
                "sort_by": {"type": "string", "enum": ["score_desc", "score_asc", "newest", "oldest"], "description": "Ordering before truncating to `limit`. Default score_desc (most positive first)."},
                "limit": {"type": "integer", "description": "Max results to return. Default 10, max 50."},
            },
        },
    },
    {
        "name": "get_stats",
        "description": (
            "Get aggregate counts and average sentiment grouped by a dimension, computed over "
            "the FULL dataset (not a sample). Use this for questions about breakdowns or totals "
            "rather than guessing from the summary numbers you were given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["source", "badge", "sentiment_label", "topic", "hour_of_day", "author"],
                    "description": "Dimension to group by. 'author' returns the top 20 most active authors.",
                },
            },
            "required": ["group_by"],
        },
    },
]

def _run_data_tool(df: pd.DataFrame, name: str, tool_input: dict) -> dict:
    try:
        if name == "search_messages":
            sub = df
            if tool_input.get("keyword"):
                sub = sub[sub["text"].str.contains(re.escape(tool_input["keyword"]), case=False, na=False)]
            if tool_input.get("sentiment"):
                sub = sub[sub["sentiment_label"] == tool_input["sentiment"]]
            if tool_input.get("source"):
                sub = sub[sub["source"] == tool_input["source"]]
            if tool_input.get("topic") and "topic" in sub.columns:
                sub = sub[sub["topic"].str.casefold() == str(tool_input["topic"]).casefold()]
            if tool_input.get("author"):
                sub = sub[sub["author"].str.contains(re.escape(tool_input["author"]), case=False, na=False)]
            total = int(len(sub))
            sort_by = tool_input.get("sort_by", "score_desc")
            if sort_by == "score_desc":
                sub = sub.sort_values("sentiment_score", ascending=False)
            elif sort_by == "score_asc":
                sub = sub.sort_values("sentiment_score", ascending=True)
            elif sort_by == "newest":
                sub = sub.sort_values("timestamp", ascending=False)
            elif sort_by == "oldest":
                sub = sub.sort_values("timestamp", ascending=True)
            limit = max(1, min(int(tool_input.get("limit", 10) or 10), 50))
            cols = [c for c in ["source", "author", "text", "sentiment_label", "sentiment_score", "topic"] if c in sub.columns]
            rows = sub.head(limit)[cols].to_dict("records")
            return {"total_matches": total, "results": rows}

        elif name == "get_stats":
            group_by = tool_input.get("group_by")
            if group_by == "hour_of_day":
                if "timestamp" not in df.columns or not df["timestamp"].notna().any():
                    return {"error": "No usable timestamps in this dataset."}
                tmp = df.dropna(subset=["timestamp"]).copy()
                tmp["hour_of_day"] = tmp["timestamp"].dt.hour
                g = tmp.groupby("hour_of_day").agg(
                    count=("text", "size"), avg_sentiment=("sentiment_score", "mean")
                ).reset_index()
                return {"rows": g.to_dict("records")}
            elif group_by == "author":
                g = df["author"].value_counts().head(20).reset_index()
                g.columns = ["author", "count"]
                return {"rows": g.to_dict("records")}
            elif group_by in ("source", "badge", "sentiment_label", "topic"):
                if group_by not in df.columns:
                    return {"error": f"'{group_by}' not available for this dataset."}
                g = df.groupby(group_by).agg(
                    count=("text", "size"), avg_sentiment=("sentiment_score", "mean")
                ).reset_index()
                return {"rows": g.to_dict("records")}
            return {"error": f"Unknown group_by: {group_by}"}

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"Tool call failed: {e}"}

def _build_chat_system_prompt(df: pd.DataFrame, video_id: str, scope: str) -> str:
    total = len(df)
    pos = int((df["sentiment_label"] == "Positive").sum())
    neu = int((df["sentiment_label"] == "Neutral").sum())
    neg = int((df["sentiment_label"] == "Negative").sum())
    kw = top_keywords(df["text"].tolist(), n=15)

    parts = [
        f"You are a helpful analyst chatting with someone about YouTube audience reaction data "
        f"for video `{video_id}` (scope: {scope}).",
        f"Dataset totals: {total:,} messages, {df['author'].nunique():,} unique authors, "
        f"{pos:,} Positive / {neu:,} Neutral / {neg:,} Negative, "
        f"average sentiment score {df['sentiment_score'].mean():+.3f} (range -1 to 1).",
    ]
    if kw:
        parts.append("Top keywords across the dataset: " + ", ".join(f"{w} ({c})" for w, c in kw))
    if "topic" in df.columns:
        topic_counts = df["topic"].value_counts()
        parts.append("Topic breakdown (every message was tagged with one of these): " +
                      ", ".join(f"{t} ({c})" for t, c in topic_counts.items()))
    ai_summary = st.session_state.get("ai_summary")
    if ai_summary:
        parts.append("A theme summary was already generated for this dataset:\n" + ai_summary)
    parts.append(
        "You have two tools: search_messages (search/filter/sort the FULL set of real messages) "
        "and get_stats (aggregate counts/averages by a dimension, over the full dataset). Use them "
        "whenever the person asks for specific examples, quotes, counts, or breakdowns — never "
        "invent a quote or a number you haven't pulled from a tool call. Keep answers conversational "
        "and reasonably short; pull in a couple of real quotes via search_messages when they'd help."
    )
    return "\n\n".join(parts)

def chat_with_data(df: pd.DataFrame, video_id: str, scope: str, user_message: str) -> str:
    """One turn of the chat-with-data assistant. Maintains history in
    st.session_state['chat_messages'] (raw Anthropic message dicts, echoed back
    across turns so multi-turn tool use and follow-up questions both work)."""
    client = get_bedrock_client()
    system = _build_chat_system_prompt(df, video_id, scope)
    messages = list(st.session_state.get("chat_messages", []))
    messages.append({"role": "user", "content": user_message})

    for _ in range(CHAT_MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=SUMMARY_MODEL_ID,
            max_tokens=CHAT_MAX_TOKENS,
            system=system,
            messages=messages,
            tools=DATA_CHAT_TOOLS,
        )
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            st.session_state["chat_messages"] = messages
            return text or "(no response text)"

        tool_results = []
        for tu in tool_uses:
            result = _run_data_tool(df, tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    st.session_state["chat_messages"] = messages
    return "That took more tool calls than I'd expect — could you try rephrasing the question?"

# ─────────────────────────────────────────────────────────────────────────────
# PDF report generation
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_png_bytes(fig, width=720, height=380):
    return fig.to_image(format="png", width=width, height=height, scale=2)

def build_pdf_report(meta: dict, df: pd.DataFrame, ai_summary: str, figs: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, Image, PageBreak)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#111111"))
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=colors.HexColor("#222222"),
                         spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], leading=15)
    meta_style = ParagraphStyle("MetaX", parent=styles["Normal"], textColor=colors.HexColor("#666666"))

    story = [
        Paragraph("YouTube Sentiment Report", title_style),
        Paragraph(
            f"Video ID: {meta.get('video_id', 'N/A')} &nbsp;|&nbsp; "
            f"Scope: {meta.get('scope')} &nbsp;|&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            meta_style,
        ),
        Spacer(1, 14),
    ]

    # KPI table
    total = len(df)
    pos = (df["sentiment_label"] == "Positive").sum()
    neu = (df["sentiment_label"] == "Neutral").sum()
    neg = (df["sentiment_label"] == "Negative").sum()
    avg = df["sentiment_score"].mean() if total else 0
    kpi_data = [
        ["Total Messages", "Unique Authors", "Avg. Sentiment", "% Positive", "% Negative"],
        [f"{total:,}", f"{df['author'].nunique():,}", f"{avg:+.3f}",
         f"{(pos/total*100 if total else 0):.1f}%", f"{(neg/total*100 if total else 0):.1f}%"],
    ]
    kpi_table = Table(kpi_data, hAlign="LEFT", colWidths=[1.4 * inch] * 5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [kpi_table, Spacer(1, 16)]

    # Charts
    for title, fig in figs.items():
        try:
            img_bytes = _fig_to_png_bytes(fig)
            story.append(Paragraph(title, h2))
            story.append(Image(io.BytesIO(img_bytes), width=6.4 * inch, height=3.4 * inch))
            story.append(Spacer(1, 8))
        except Exception:
            continue  # kaleido not available — skip chart, report still generates

    story.append(PageBreak())

    # AI summary
    story.append(Paragraph("AI-Generated Insight Summary", h2))
    for line in ai_summary.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        clean = line.lstrip("#").strip()
        if line.startswith("#") or (line.endswith(":") and len(line) < 60):
            story.append(Paragraph(f"<b>{clean}</b>", body))
        else:
            story.append(Paragraph(clean.replace("**", ""), body))

    story.append(PageBreak())

    # Top positive / negative quotes
    story.append(Paragraph("Notable Messages", h2))
    top_pos = df.sort_values("sentiment_score", ascending=False).head(5)
    top_neg = df.sort_values("sentiment_score", ascending=True).head(5)

    story.append(Paragraph("<b>Most Positive</b>", body))
    for r in top_pos.itertuples():
        story.append(Paragraph(f"({r.sentiment_score:+.2f}) <b>{r.author}</b>: {r.text}", body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Most Negative</b>", body))
    for r in top_neg.itertuples():
        story.append(Paragraph(f"({r.sentiment_score:+.2f}) <b>{r.author}</b>: {r.text}", body))

    doc.build(story)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### 📊 Stream Sentiment Dashboard")
st.sidebar.caption(f"Signed in as {st.session_state.get('auth_email', '')}")
st.sidebar.divider()

platform = st.sidebar.radio("Platform", ["YouTube", "Twitch"], horizontal=True)
st.sidebar.divider()

if platform == "YouTube":
    scope = st.sidebar.radio(
        "Analyze",
        ["Chat only", "Comments only", "Both"],
        help="Live chat / replay uses pytchat. Comments use youtube-comment-downloader.",
    )
    need_chat = scope in ("Chat only", "Both")
    need_comments = scope in ("Comments only", "Both")
else:
    # Twitch has no separate "comments" section the way YouTube videos do —
    # it's chat (live tail or VOD/clip replay) only.
    scope = "Chat only"
    need_chat = True
    need_comments = False
    st.sidebar.caption("Twitch: chat only — there's no separate comments section to analyze.")

st.sidebar.divider()
st.sidebar.markdown("#### Data source")
source_mode = st.sidebar.radio("Get data via", ["Video ID (fetch live)", "Upload JSON"])

video_id = None
chat_upload = None
comments_upload = None

if source_mode == "Video ID (fetch live)":
    if platform == "YouTube":
        video_id = st.sidebar.text_input("YouTube Video ID", placeholder="e.g. eVq6qlu0_GU")
    else:
        video_id = st.sidebar.text_input(
            "Twitch channel or VOD/clip URL",
            placeholder="twitch.tv/somechannel  or  twitch.tv/videos/123456789",
            help="A channel name or channel URL tails live chat in real time. A "
                 "twitch.tv/videos/<id> or clips.twitch.tv URL instead fetches archived "
                 "VOD/clip chat replay — auto-detected from what you paste.",
        )

    with st.sidebar.expander("Fetch limits (optional)", expanded=False):
        st.caption("Off by default — every message/comment is fetched. Only turn these on "
                   "if you want to bound a very large or currently-live stream.")

        if platform == "YouTube":
            comment_sort = st.selectbox("Comment sort", ["Popular", "Recent"])
        else:
            comment_sort = None

        cap_chat = st.checkbox("Cap chat message count", value=False)
        max_chat_messages = st.number_input("Max chat messages", 100, 500000, 3000, step=100,
                                             disabled=not cap_chat)
        chat_timeout_min = st.number_input(
            "Chat safety timeout (minutes)", 1, 720, 60,
            help=("Only matters for a currently-live stream/channel, which polls forever until "
                  "it ends. An archived VOD/replay finishes on its own regardless of this."),
        )

        if platform == "YouTube":
            cap_comments = st.checkbox("Cap comment count", value=False)
            max_comments = st.number_input("Max comments", 50, 500000, 1000, step=50,
                                            disabled=not cap_comments)
            st.divider()
            topchat_only = st.checkbox(
                "Top chat only (faster, less complete)", value=False,
                help="Native pytchat option that returns only YouTube's highlighted 'top chat' "
                     "messages instead of every message — meaningfully fewer network round-trips "
                     "on very high-traffic streams, at the cost of completeness.",
            )
        else:
            cap_comments = False
            max_comments = None
            topchat_only = False
else:
    if need_chat:
        chat_upload = st.sidebar.file_uploader("Chat JSON", type=["json"], key="chat_json")
    if need_comments:
        comments_upload = st.sidebar.file_uploader("Comments JSON", type=["json"], key="comments_json")

with st.sidebar.expander("Sentiment scoring (Claude Haiku)", expanded=False):
    st.caption("Every message is classified by Claude Haiku via Bedrock, in concurrent "
               "batches — no VADER lexicon involved. All items are analyzed by default.")
    cap_analyze = st.checkbox("Cap items analyzed per uploaded JSON", value=False)
    max_analyze_items = st.number_input(
        "Max items to analyze per uploaded JSON", 100, 500000, 3000, step=100,
        disabled=not cap_analyze,
        help="Off by default — the full file is scored. Turn this on to bound cost/time "
             "on a very large export.",
    )

with st.sidebar.expander("Topic categories", expanded=False):
    st.caption("Every message also gets tagged with a topic. \"Other\" is always added "
               "automatically as a catch-all for anything that doesn't fit.")
    topic_mode = st.radio("How to choose categories", ["Type my own", "Auto-detect from data"],
                           horizontal=True)
    auto_detect_topics = topic_mode == "Auto-detect from data"
    if auto_detect_topics:
        st.caption(
            "Claude looks at a sample of the fetched/uploaded text and proposes categories "
            "before scoring starts — one extra Bedrock call. Trade-off: this run can't overlap "
            "fetching with scoring (it needs to see the data first), so it'll be somewhat slower "
            "than a manual-category run on the same dataset. Detected categories are shown after "
            "the run so you can reuse them as a manual list next time if you want the speed back."
        )
        topic_category_range = st.slider(
            "How many categories to detect", min_value=2, max_value=12, value=(4, 6),
            help="Claude proposes a category count in this range — it'll use the low end if the "
                 "content is genuinely narrow rather than padding the list to hit the top end.",
        )
        topic_categories_input = None
    else:
        topic_categories_input = st.text_input(
            "Categories (comma-separated)",
            value="Games, Characters, Features, Music, Requests",
        )
        st.caption("Same Haiku pass also tags each message with one of these, so it costs no "
                   "extra API calls.")

if auto_detect_topics:
    topic_categories = None  # resolved per-run in load_data() from a sample of the actual data
else:
    topic_categories = _normalize_topic_categories(
        [c.strip() for c in topic_categories_input.split(",") if c.strip()]
    )

run_ai_summary = st.sidebar.checkbox("Generate AI theme summary (Bedrock)", value=True)

st.sidebar.divider()
run_clicked = st.sidebar.button("▶ Run Analysis", width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — build the working dataset
# ─────────────────────────────────────────────────────────────────────────────

def _throttle(fn, min_interval=0.15):
    """Wrap a progress callback so it actually touches the Streamlit UI at
    most every `min_interval` seconds, no matter how often it's called.

    Every widget update (st.progress/.text/...) costs real time — it builds
    and pushes a delta to the frontend, not just a Python function call — so
    firing one per message on a large fetch adds up: even in a lightweight
    test harness, 5,000 raw placeholder updates took ~0.8s on their own, and
    a real browser session over a websocket is typically slower. Since the
    placeholder gets cleared right after the fetch/score loop finishes
    either way, skipping intermediate updates between the throttle window is
    harmless — the person just sees the counter tick up in ~150ms steps
    instead of on every single item."""
    state = {"last": 0.0}
    def wrapped(*args, **kwargs):
        now = time.time()
        if now - state["last"] >= min_interval:
            state["last"] = now
            fn(*args, **kwargs)
    return wrapped

def _scoring_progress(label):
    progress = st.progress(0.0, text=f"Scoring {label} sentiment with Claude Haiku...")
    def cb(done, total):
        progress.progress(min(done / total, 1.0) if total else 1.0,
                           text=f"Scoring {label} sentiment with Claude Haiku... ({done}/{total})")
        if done >= total:
            progress.empty()
    return _throttle(cb)

def _fetch_chat_source(platform, identifier, cap, timeout_seconds, progress_cb, topchat_only, scorer=None):
    """Dispatch to the right chat fetcher for the selected platform. Both
    return the same item schema, so everything downstream (chat_to_dataframe,
    StreamingScorer, etc.) is completely platform-agnostic."""
    if platform == "YouTube":
        return fetch_live_chat(identifier, cap, timeout_seconds, progress_cb=progress_cb,
                                topchat_only=topchat_only, scorer=scorer)
    try:
        return fetch_twitch_chat(identifier, cap, timeout_seconds, progress_cb=progress_cb, scorer=scorer)
    except TwitchChatError as e:
        st.error(str(e))
        st.stop()

def load_data():
    chat_items, comments_items = None, None
    chat_scores, chat_failed = None, 0
    comments_scores, comments_failed = None, 0
    chat_df, comments_df = pd.DataFrame(), pd.DataFrame()

    # ── Chat: fetch/load raw items. Scored inline (via StreamingScorer, overlapped
    #    with fetching) unless auto-detect is on, in which case scoring is deferred
    #    until categories are resolved from a sample below. ──────────────────────
    if need_chat:
        if source_mode == "Video ID (fetch live)":
            if not video_id:
                st.error("Enter a video ID to fetch chat." if platform == "YouTube"
                          else "Enter a Twitch channel name or VOD/clip URL to fetch chat.")
                st.stop()
            fetch_status = st.empty()
            chat_cap = int(max_chat_messages) if cap_chat else None
            if cap_chat:
                def cb(n):
                    fetch_status.progress(min(n / chat_cap, 1.0), text=f"Fetched {n:,} chat messages...")
            else:
                def cb(n):
                    fetch_status.text(f"Fetched {n:,} chat messages so far...")
            cb = _throttle(cb)

            if auto_detect_topics:
                with st.spinner("Fetching live chat / replay..."):
                    chat_items = _fetch_chat_source(platform, video_id, chat_cap,
                                                     int(chat_timeout_min) * 60, cb, topchat_only)
            else:
                score_status_cb = _scoring_progress("chat")
                # Scoring runs concurrently with fetching via StreamingScorer — Haiku
                # batches get submitted as soon as they fill, while the fetch loop
                # keeps pulling more pages in the background thread pool's downtime.
                scorer = StreamingScorer(status_cb=score_status_cb, topic_categories=topic_categories)
                with st.spinner("Fetching live chat / replay (scoring runs alongside it)..."):
                    chat_items = _fetch_chat_source(platform, video_id, chat_cap,
                                                     int(chat_timeout_min) * 60, cb, topchat_only,
                                                     scorer=scorer)
                    chat_scores, chat_failed = scorer.finish()
            fetch_status.empty()
            if not chat_items:
                st.warning("No chat messages retrieved. Check the video ID and that chat/replay is enabled."
                            if platform == "YouTube" else
                            "No chat messages retrieved. Check the channel/VOD URL and that chat replay "
                            "is available.")
        else:
            if not chat_upload:
                st.error("Upload a chat JSON file.")
                st.stop()
            chat_items = load_chat_json(chat_upload.read())
            if cap_analyze and len(chat_items) > max_analyze_items:
                st.info(f"Chat JSON has {len(chat_items):,} messages — analyzing the first "
                         f"{max_analyze_items:,} (raise or disable the cap in the sidebar to "
                         f"analyze all of it).")
                chat_items = chat_items[:max_analyze_items]

    # ── Comments: same pattern ──────────────────────────────────────────────────
    if need_comments:
        if source_mode == "Video ID (fetch live)":
            if not video_id:
                st.error("Enter a video ID to fetch comments.")
                st.stop()
            fetch_status2 = st.empty()
            comments_cap = int(max_comments) if cap_comments else None
            if cap_comments:
                def cb2(n):
                    fetch_status2.progress(min(n / comments_cap, 1.0), text=f"Fetched {n:,} comments...")
            else:
                def cb2(n):
                    fetch_status2.text(f"Fetched {n:,} comments so far...")
            cb2 = _throttle(cb2)

            if auto_detect_topics:
                with st.spinner("Fetching comments..."):
                    comments_items = fetch_comments(video_id, comments_cap, comment_sort, progress_cb=cb2)
            else:
                score_status_cb2 = _scoring_progress("comment")
                scorer2 = StreamingScorer(status_cb=score_status_cb2, topic_categories=topic_categories)
                with st.spinner("Fetching comments (scoring runs alongside it)..."):
                    comments_items = fetch_comments(video_id, comments_cap, comment_sort,
                                                     progress_cb=cb2, scorer=scorer2)
                    comments_scores, comments_failed = scorer2.finish()
            fetch_status2.empty()
            if not comments_items:
                st.warning("No comments retrieved. Check the video ID and that comments are enabled.")
        else:
            if not comments_upload:
                st.error("Upload a comments JSON file.")
                st.stop()
            comments_items = load_comments_json(comments_upload.read())
            if cap_analyze and len(comments_items) > max_analyze_items:
                st.info(f"Comments JSON has {len(comments_items):,} items — analyzing the first "
                         f"{max_analyze_items:,} (raise or disable the cap in the sidebar to "
                         f"analyze all of it).")
                comments_items = comments_items[:max_analyze_items]

    # ── Resolve topic categories. For auto-detect, this is where the extra
    #    Bedrock call happens — on one combined sample across whichever sources
    #    were fetched, so chat and comments share a single taxonomy instead of
    #    getting two different ones for "Both" scope. ───────────────────────────
    resolved_categories = topic_categories
    if auto_detect_topics:
        combined_texts = []
        if chat_items:
            combined_texts += [it.get("message") for it in chat_items]
        if comments_items:
            combined_texts += [it.get("text") for it in comments_items]
        with st.spinner("Auto-detecting topic categories from a sample of the data..."):
            resolved_categories = discover_topic_categories(
                combined_texts,
                min_categories=topic_category_range[0],
                max_categories=topic_category_range[1],
            )
        st.session_state["auto_detected_topics"] = resolved_categories
        st.success(f"Auto-detected topics: {', '.join(resolved_categories)} (+ Other)")

    # ── Score whatever wasn't already scored inline above ───────────────────────
    if chat_items:
        if chat_scores is not None:
            chat_df = chat_to_dataframe(chat_items, precomputed_scores=chat_scores)
        else:
            st.caption(f"Scoring sentiment on all {len(chat_items):,} chat messages...")
            with st.spinner("Scoring sentiment with Claude Haiku..."):
                chat_df = chat_to_dataframe(chat_items, status_cb=_scoring_progress("chat"),
                                             topic_categories=resolved_categories)
            chat_failed = st.session_state.pop("_sentiment_failed_batches", 0)
        if chat_failed:
            st.warning("Some chat message batches couldn't be scored and were marked Neutral "
                        "(model call failed after retries). Re-run if this seems off.")

    if comments_items:
        if comments_scores is not None:
            comments_df = comments_to_dataframe(comments_items, precomputed_scores=comments_scores)
        else:
            st.caption(f"Scoring sentiment on all {len(comments_items):,} comments...")
            with st.spinner("Scoring sentiment with Claude Haiku..."):
                comments_df = comments_to_dataframe(comments_items, status_cb=_scoring_progress("comment"),
                                                     topic_categories=resolved_categories)
            comments_failed = st.session_state.pop("_sentiment_failed_batches", 0)
        if comments_failed:
            st.warning("Some comment batches couldn't be scored and were marked Neutral "
                        "(model call failed after retries). Re-run if this seems off.")

    return chat_df, comments_df

# ─────────────────────────────────────────────────────────────────────────────
# Bundled default dataset — lets the app open with the team's primary dataset
# already loaded instead of an empty state, and skips re-running (slow,
# costly) analysis on a 77k-message dataset that's already been scored once.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "default_dataset.csv"
DEFAULT_DATASET_LABEL = "Default dataset (bundled)"

@st.cache_data(show_spinner=False)
def load_bundled_default_dataset():
    """Load the pre-scored CSV bundled alongside this script — same schema as
    the app's own CSV export (source/text/author/timestamp/sentiment_score/
    sentiment_label/topic/badge). Returns (chat_df, comments_df, error).
    error is None either on success OR when the file is simply absent (the
    normal, expected case if no bundle was deployed) — it's only set when the
    file EXISTS but something about it is wrong, so that genuine problem is
    distinguishable in the UI from "no bundle configured" instead of both
    cases silently looking identical."""
    if not DEFAULT_DATASET_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), None

    try:
        df = pd.read_csv(DEFAULT_DATASET_PATH)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), (
            f"Found {DEFAULT_DATASET_PATH.name} at {DEFAULT_DATASET_PATH.parent} but couldn't "
            f"parse it as CSV: {e}"
        )

    required_cols = {"source", "text", "sentiment_score", "sentiment_label"}
    missing = required_cols - set(df.columns)
    if missing:
        return pd.DataFrame(), pd.DataFrame(), (
            f"Found {DEFAULT_DATASET_PATH.name} but it's missing expected column(s): "
            f"{', '.join(sorted(missing))}. Expected the app's own CSV export schema "
            f"(source/text/author/timestamp/sentiment_score/sentiment_label/topic/badge)."
        )

    df["text"] = df.get("text", pd.Series(dtype=str)).fillna("")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    source_col = df["source"] if "source" in df.columns else pd.Series(["Chat"] * len(df))
    chat_df = df[source_col == "Chat"].reset_index(drop=True)
    comments_df = df[source_col == "Comment"].reset_index(drop=True)

    if chat_df.empty and comments_df.empty:
        return pd.DataFrame(), pd.DataFrame(), (
            f"Found {DEFAULT_DATASET_PATH.name} but it parsed to 0 usable rows — check that its "
            f"'source' column contains 'Chat' and/or 'Comment' values."
        )

    return chat_df, comments_df, None

if run_clicked:
    st.session_state["auto_detected_topics"] = None  # reset before load_data() may repopulate it
    chat_df, comments_df = load_data()
    st.session_state["chat_df"] = chat_df
    st.session_state["comments_df"] = comments_df
    st.session_state["scope"] = scope
    st.session_state["platform"] = platform
    st.session_state["video_id"] = video_id or "(uploaded JSON)"
    st.session_state["ai_summary"] = None  # reset any prior summary
    st.session_state["chat_messages"] = []  # reset chat-with-data history for the new dataset
    st.session_state["chat_display"] = []
    st.session_state["segment_ai_titles"] = None  # reset any prior AI segment titles
    st.session_state["segment_ai_titles_freq"] = None
    st.session_state["is_default_dataset"] = False
    st.session_state["data_loaded"] = True
elif "data_loaded" not in st.session_state:
    # First time this session has rendered at all — try the bundled default
    # dataset instead of showing an empty "configure a source" screen. Only
    # runs once per session: after this, data_loaded is always set, so later
    # reruns (widget interactions, a real Run Analysis) never touch this again.
    default_chat_df, default_comments_df, default_error = load_bundled_default_dataset()
    if default_error:
        st.session_state["default_dataset_error"] = default_error
    if not default_chat_df.empty or not default_comments_df.empty:
        st.session_state["chat_df"] = default_chat_df
        st.session_state["comments_df"] = default_comments_df
        if not default_chat_df.empty and not default_comments_df.empty:
            st.session_state["scope"] = "Both"
        elif not default_chat_df.empty:
            st.session_state["scope"] = "Chat only"
        else:
            st.session_state["scope"] = "Comments only"
        st.session_state["video_id"] = DEFAULT_DATASET_LABEL
        st.session_state["platform"] = "YouTube"
        st.session_state["is_default_dataset"] = True
        st.session_state["data_loaded"] = True

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {"Positive": "#2fbf71", "Neutral": "#8a8a8a", "Negative": "#d92b2b"}

def combined_df():
    chat_df = st.session_state.get("chat_df", pd.DataFrame())
    comments_df = st.session_state.get("comments_df", pd.DataFrame())
    frames = [d for d in (chat_df, comments_df) if not d.empty]
    if not frames:
        return pd.DataFrame()
    cols = ["source", "text", "author", "timestamp", "sentiment_score", "sentiment_label", "topic", "badge"]
    frames = [d[[c for c in cols if c in d.columns]] for d in frames]
    return pd.concat(frames, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat Overview tab
# ─────────────────────────────────────────────────────────────────────────────

def _compute_activity_stats(df: pd.DataFrame):
    """Average messages/min and peak 1-minute activity, or None if there
    aren't enough usable/distinct timestamps to make rate metrics meaningful."""
    if "timestamp" not in df.columns:
        return None
    ts = df.dropna(subset=["timestamp"])
    if len(ts) < 2 or ts["timestamp"].nunique() < 2:
        return None
    duration_min = (ts["timestamp"].max() - ts["timestamp"].min()).total_seconds() / 60
    if duration_min <= 0:
        return None
    per_min = ts.set_index("timestamp").resample("1min").size()
    peak_count = int(per_min.max())
    peak_time = per_min.idxmax()
    return {
        "avg_rate": len(ts) / duration_min,
        "peak_count": peak_count,
        "peak_time": peak_time,
        "duration_min": duration_min,
        "coarse": ts["timestamp"].nunique() < max(10, len(ts) * 0.1),
    }

def render_chat_overview_tab(df: pd.DataFrame):
    total = len(df)
    unique_users = df["author"].nunique()
    stats = _compute_activity_stats(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", f"{total:,}")
    c2.metric("Unique Users", f"{unique_users:,}")
    if stats:
        c3.metric("Avg. Messages / Min", f"{stats['avg_rate']:.1f}")
        c4.metric("Peak Activity", f"{stats['peak_count']:,} / min",
                   help=f"at {stats['peak_time'].strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        c3.metric("Avg. Messages / Min", "n/a")
        c4.metric("Peak Activity", "n/a")

    if stats and stats["coarse"]:
        st.caption("⚠️ Timestamps in this dataset look coarse (lots of duplicate/rounded values — "
                   "common for older comments), so rate metrics above may be rough approximations.")
    elif not stats:
        st.caption("Not enough usable timestamps in this dataset to compute rate metrics.")

    st.divider()
    col1, col2 = st.columns([1, 1.4])

    with col1:
        src_counts = df["source"].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        if len(src_counts) > 1:
            fig = go.Figure(data=[go.Pie(
                labels=src_counts["source"], values=src_counts["count"], hole=0.55,
                marker=dict(colors=["#5c9eff", "#f5c518"]),
                textinfo="label+percent",
            )])
            fig.update_layout(
                title="Live Chat vs. Comments (volume)", template="plotly_dark",
                paper_bgcolor="#080808", plot_bgcolor="#080808", height=380,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            only = src_counts.iloc[0]
            st.metric(f"{only['source']} messages", f"{int(only['count']):,}")
            st.caption("Only one source was analyzed this run, so there's nothing to compare here.")

    with col2:
        if stats:
            ts = df.dropna(subset=["timestamp"])
            freq = st.select_slider("Volume bucket size", options=["1min", "5min", "15min", "1H"],
                                     value="5min", key="chatoverview_bucket")
            vol = (
                ts.set_index("timestamp")
                .groupby([pd.Grouper(freq=freq), "source"])
                .size()
                .reset_index(name="count")
            )
            fig2 = px.line(vol, x="timestamp", y="count", color="source", title="Message Volume Over Time")
            fig2.update_layout(
                template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                height=380, margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig2, width='stretch')
        else:
            st.caption("Volume-over-time needs usable timestamps, which this dataset doesn't have.")

# ─────────────────────────────────────────────────────────────────────────────
# Topic Analysis tab
# ─────────────────────────────────────────────────────────────────────────────

def render_topic_analysis_tab(df: pd.DataFrame, figs_for_report: dict):
    if "topic" not in df.columns:
        st.info("No topic data available for this dataset.")
        return

    auto_detected = st.session_state.get("auto_detected_topics")
    if auto_detected:
        st.caption(f"🔍 Categories were auto-detected from this data: {', '.join(auto_detected)} "
                   "(+ Other). Paste these into the sidebar's \"Type my own\" list to reuse them "
                   "on a re-run with the fetch/score overlap back on.")

    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]

    col1, col2 = st.columns(2)
    with col1:
        plot_df = topic_counts.sort_values("count", ascending=True)
        fig_topic = px.bar(plot_df, x="count", y="topic", orientation="h", title="Topic Distribution")
        fig_topic.update_traces(marker_color="#b06fe0")
        fig_topic.update_layout(
            template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
            height=420, margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_topic, width='stretch')
        figs_for_report["Topic Distribution"] = fig_topic

    with col2:
        by_topic_sent = df.groupby(["topic", "sentiment_label"]).size().reset_index(name="count")
        fig_ts = px.bar(
            by_topic_sent, x="topic", y="count", color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS, barmode="stack", title="Sentiment by Topic",
        )
        fig_ts.update_layout(
            template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
            height=420, margin=dict(t=50, b=10, l=10, r=10), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_ts, width='stretch')
        figs_for_report["Sentiment by Topic"] = fig_ts

    st.divider()
    st.markdown("#### Example messages by topic")
    topics_ordered = [t for t in topic_counts.sort_values("count", ascending=False)["topic"] if t != OTHER_TOPIC]
    if not topics_ordered:
        st.caption(f"Every message landed in \"{OTHER_TOPIC}\" — try adjusting the topic categories "
                   "in the sidebar (under Topic categories) and re-running.")
    for t in topics_ordered:
        sub = df[df["topic"] == t]
        with st.expander(f"{t}  ·  {len(sub):,} messages  ·  avg sentiment {sub['sentiment_score'].mean():+.2f}"):
            sample = sub.reindex(sub["sentiment_score"].abs().sort_values(ascending=False).index).head(5)
            for r in sample.itertuples():
                cls = "quote-pos" if r.sentiment_score > 0 else ("quote-neg" if r.sentiment_score < 0 else "")
                st.markdown(
                    f'<div class="quote-card {cls}"><div class="quote-meta">{r.source} · {r.author} · '
                    f'{r.sentiment_label} ({r.sentiment_score:+.2f})</div>'
                    f'<div class="quote-text">{r.text}</div></div>', unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Segments tab — approximate stream segments from chat topic over time
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_topics(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """One row per non-empty time bucket: dominant topic, message count, avg sentiment.

    Dominant topic prefers the strongest non-"Other" topic in the bucket, only
    falling back to "Other" if the bucket has no other signal at all. "Other"
    is a generic catch-all (greetings, hype, spam) that's very often the single
    largest bucket by plain plurality — sometimes barely over a third of
    messages — so using a plain mode() lets "Other" win almost every bucket and
    swallows real topic shifts into one giant merged segment. Excluding it from
    the vote (except as a last resort) surfaces the topic that's actually
    distinctive about that window instead."""
    ts = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if ts.empty:
        return pd.DataFrame()
    rows = []
    for bucket_start, sub in ts.set_index("timestamp").groupby(pd.Grouper(freq=freq)):
        if len(sub) == 0:
            continue
        non_other = sub[sub["topic"] != OTHER_TOPIC]
        if len(non_other) > 0:
            dominant = non_other["topic"].mode().iloc[0]
        else:
            dominant = OTHER_TOPIC
        rows.append({
            "bucket_start": bucket_start,
            "dominant_topic": dominant,
            "count": len(sub),
            "avg_sentiment": sub["sentiment_score"].mean(),
        })
    return pd.DataFrame(rows)

def _merge_into_segments(bucket_df: pd.DataFrame, freq_td: pd.Timedelta) -> list:
    """Merge consecutive, time-adjacent buckets that share a dominant topic into
    segments. A real gap in activity (a skipped/empty bucket) breaks the merge
    even if the topic on both sides matches, so silent gaps aren't papered over."""
    segments = []
    current = None
    for row in bucket_df.itertuples():
        if (current is not None and row.dominant_topic == current["topic"]
                and (row.bucket_start - current["end"]) <= freq_td):
            current["end"] = row.bucket_start + freq_td
            current["count"] += row.count
            current["sent_sum"] += row.avg_sentiment * row.count
        else:
            if current is not None:
                segments.append(current)
            current = {
                "topic": row.dominant_topic, "start": row.bucket_start, "end": row.bucket_start + freq_td,
                "count": row.count, "sent_sum": row.avg_sentiment * row.count,
            }
    if current is not None:
        segments.append(current)
    for s in segments:
        s["avg_sentiment"] = s["sent_sum"] / s["count"] if s["count"] else 0.0
    return segments

def generate_segment_titles(segments_info: list, video_id: str) -> list:
    """One Bedrock call covering every segment at once: a short title + one-sentence
    guess at what was happening on stream, grounded in that segment's dominant
    topic/keywords/quotes. Segments come from chat alone, so this is an inference,
    not a transcript — the prompt says so explicitly."""
    lines = []
    for s in segments_info:
        lines.append(
            f"Segment {s['idx']} ({s['start']}\u2013{s['end']}, {s['count']} messages, "
            f"dominant topic: {s['topic']}, avg sentiment {s['avg_sentiment']:+.2f}):\n"
            f"  top keywords: {', '.join(s['keywords']) or 'n/a'}\n"
            f"  sample messages: " + " | ".join(q[:150] for q in s['quotes'])
        )
    system = (
        "You are labeling segments of a YouTube livestream based on chat/comment activity alone — "
        "there's no video or audio available, so your labels are an approximation from chat content. "
        "For each segment, write a short punchy title (a few words) and a one-sentence description "
        "of what was probably happening on stream, grounded in the topic/keywords/quotes given. "
        "Don't overclaim certainty since this is inferred from chat, not confirmed."
    )
    user = (
        f"Video `{video_id}` — chat-derived segments, in chronological order:\n\n"
        + "\n\n".join(lines)
        + "\n\nRespond with ONLY a JSON array, no prose, no markdown fences: "
          '[{"idx": <int>, "title": "<short title>", "description": "<one sentence>"}, ...]'
    )
    raw = _bedrock_call(system, user, max_tokens=1500)
    return _parse_json_array(raw)

def render_segments_tab(df: pd.DataFrame, video_id: str):
    ts_df = df.dropna(subset=["timestamp"]) if "timestamp" in df.columns else pd.DataFrame()
    if ts_df.empty or "topic" not in df.columns:
        st.info("Segmentation needs both usable timestamps and topic data — this dataset doesn't "
                "have enough of one or the other.")
        return

    st.caption(
        "Buckets the timeline by dominant chat topic to approximate what the stream might have "
        "been covering at each point. This is inferred purely from chat/comment content, not the "
        "actual video or audio — treat it as a best guess, not a transcript."
    )

    freq = st.select_slider("Segment bucket size", options=["1min", "2min", "5min", "10min", "15min"],
                             value="5min", key="segment_bucket")
    freq_td = pd.Timedelta(freq)

    bucket_df = _bucket_topics(df, freq)
    if bucket_df.empty:
        st.info("Not enough timestamped messages to build segments.")
        return

    segments = _merge_into_segments(bucket_df, freq_td)
    if not segments:
        st.info("Not enough data to form segments.")
        return

    segments_info = []
    for i, seg in enumerate(segments):
        sub = df[(df["timestamp"] >= seg["start"]) & (df["timestamp"] < seg["end"])]
        kw = [w for w, _ in top_keywords(sub["text"].tolist(), n=6)]
        top_quotes_df = sub.reindex(sub["sentiment_score"].abs().sort_values(ascending=False).index).head(3)
        segments_info.append({
            "idx": i, "topic": seg["topic"], "start": seg["start"], "end": seg["end"],
            "count": seg["count"], "avg_sentiment": seg["avg_sentiment"],
            "keywords": kw, "quotes": top_quotes_df["text"].tolist(),
        })

    st.caption(f"{len(segments_info)} segment(s) detected across "
               f"{ts_df['timestamp'].min().strftime('%H:%M')}\u2013{ts_df['timestamp'].max().strftime('%H:%M')}.")

    if st.session_state.get("segment_ai_titles_freq") != freq:
        st.session_state["segment_ai_titles"] = None

    if st.button("✨ Generate AI segment titles (1 Bedrock call)"):
        with st.spinner("Asking Claude to summarize each segment..."):
            try:
                fmt_segments = [
                    {**s, "start": s["start"].strftime("%H:%M"), "end": s["end"].strftime("%H:%M")}
                    for s in segments_info
                ]
                titles = generate_segment_titles(fmt_segments, video_id)
                st.session_state["segment_ai_titles"] = {t["idx"]: t for t in titles if "idx" in t}
                st.session_state["segment_ai_titles_freq"] = freq
                st.rerun()
            except Exception as e:
                st.error(f"Segment titling failed: {e}")

    ai_titles = st.session_state.get("segment_ai_titles") \
        if st.session_state.get("segment_ai_titles_freq") == freq else None

    for s in segments_info:
        time_range = f"{s['start'].strftime('%H:%M')}\u2013{s['end'].strftime('%H:%M')}"
        header = f"Segment {s['idx']+1}: {time_range} · {s['topic']} · {s['count']:,} messages · avg sentiment {s['avg_sentiment']:+.2f}"
        if ai_titles and s["idx"] in ai_titles:
            header += f" — {ai_titles[s['idx']].get('title', '')}"
        with st.expander(header):
            if ai_titles and s["idx"] in ai_titles:
                st.markdown(f"**{ai_titles[s['idx']].get('title', '')}** — "
                            f"{ai_titles[s['idx']].get('description', '')}")
            if s["keywords"]:
                st.caption("Top keywords: " + ", ".join(s["keywords"]))
            for q in s["quotes"]:
                st.markdown(f'<div class="quote-card"><div class="quote-text">{q}</div></div>',
                            unsafe_allow_html=True)

def render_dashboard():
    df = combined_df()
    if df.empty:
        default_error = st.session_state.get("default_dataset_error")
        if default_error:
            st.warning(f"⚠️ Tried to load the bundled default dataset but hit a problem: "
                       f"{default_error}")
        elif not DEFAULT_DATASET_PATH.exists():
            st.caption(f"(No bundled default dataset found at "
                      f"`{DEFAULT_DATASET_PATH}` — that's expected if none was deployed "
                      f"alongside the script.)")
        st.info("No data loaded yet. Configure a source in the sidebar and click **Run Analysis**.")
        return

    video_id = st.session_state.get("video_id", "N/A")
    scope = st.session_state.get("scope", "")

    if st.session_state.get("is_default_dataset"):
        st.info("📊 Showing the bundled default dataset — already scored, so no analysis time was "
                "needed. Configure a source in the sidebar and click **Run Analysis** to load "
                "different data.")

    st.markdown(f"## Results — `{video_id}`")
    platform_label = st.session_state.get("platform", "YouTube")
    st.caption(f"Platform: {platform_label}  •  Scope: {scope}  •  {len(df):,} total messages analyzed")

    figs_for_report = {}

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = len(df)
    pos = int((df["sentiment_label"] == "Positive").sum())
    neu = int((df["sentiment_label"] == "Neutral").sum())
    neg = int((df["sentiment_label"] == "Negative").sum())
    avg_score = df["sentiment_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Messages", f"{total:,}")
    c2.metric("Unique Authors", f"{df['author'].nunique():,}")
    c3.metric("Avg. Sentiment", f"{avg_score:+.3f}")
    c4.metric("% Positive", f"{pos/total*100:.1f}%")
    c5.metric("% Negative", f"{neg/total*100:.1f}%")

    tabs = st.tabs(["Chat Overview", "Sentiment Overview", "Timeline", "Topic Analysis", "Segments",
                    "Authors & Engagement", "Message Explorer", "AI Insights", "💬 Chat", "Export / Report"])

    # ── Chat Overview ────────────────────────────────────────────────────────
    with tabs[0]:
        render_chat_overview_tab(df)

    # ── Overview ────────────────────────────────────────────────────────────
    with tabs[1]:
        col1, col2 = st.columns([1, 1.4])

        with col1:
            dist = df["sentiment_label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
            fig_pie = go.Figure(data=[go.Pie(
                labels=dist.index, values=dist.values, hole=0.55,
                marker=dict(colors=[SENTIMENT_COLORS[l] for l in dist.index]),
                textinfo="label+percent",
            )])
            fig_pie.update_layout(
                title="Sentiment Distribution", template="plotly_dark",
                paper_bgcolor="#080808", plot_bgcolor="#080808", height=380,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, width='stretch')
            figs_for_report["Sentiment Distribution"] = fig_pie

        with col2:
            if df["source"].nunique() > 1:
                by_source = df.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
                fig_src = px.bar(
                    by_source, x="source", y="count", color="sentiment_label",
                    color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                    title="Sentiment by Source (Chat vs. Comments)",
                )
                fig_src.update_layout(
                    template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                    height=380, margin=dict(t=50, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_src, width='stretch')
                figs_for_report["Sentiment by Source"] = fig_src
            else:
                kw = top_keywords(df["text"].tolist(), n=15)
                if kw:
                    kdf = pd.DataFrame(kw, columns=["word", "count"]).sort_values("count")
                    fig_kw = px.bar(kdf, x="count", y="word", orientation="h", title="Top Keywords")
                    fig_kw.update_traces(marker_color="#d92b2b")
                    fig_kw.update_layout(
                        template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                        height=380, margin=dict(t=50, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_kw, width='stretch')
                    figs_for_report["Top Keywords"] = fig_kw

        # keyword + emoji row
        col3, col4 = st.columns(2)
        with col3:
            kw = top_keywords(df["text"].tolist(), n=15)
            if kw:
                kdf = pd.DataFrame(kw, columns=["word", "count"]).sort_values("count")
                fig_kw2 = px.bar(kdf, x="count", y="word", orientation="h", title="Top Keywords")
                fig_kw2.update_traces(marker_color="#5c9eff")
                fig_kw2.update_layout(
                    template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                    height=420, margin=dict(t=50, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_kw2, width='stretch')
        with col4:
            emoji_counter = Counter()
            for t in df["text"]:
                emoji_counter.update(extract_emojis(t))
            if emoji_counter:
                edf = pd.DataFrame(emoji_counter.most_common(15), columns=["emoji", "count"]).sort_values("count")
                fig_em = px.bar(edf, x="count", y="emoji", orientation="h", title="Top Emojis")
                fig_em.update_traces(marker_color="#f5c518")
                fig_em.update_layout(
                    template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                    height=420, margin=dict(t=50, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_em, width='stretch')
            else:
                st.caption("No emojis detected in this dataset.")

    # ── Timeline ────────────────────────────────────────────────────────────
    with tabs[2]:
        ts_df = df.dropna(subset=["timestamp"]) if "timestamp" in df.columns else pd.DataFrame()
        if ts_df.empty:
            st.info("No usable timestamps in this dataset — timeline unavailable (common for JSON uploads without `datetime`/`time_parsed` fields).")
        else:
            ts_df = ts_df.sort_values("timestamp")
            freq = st.select_slider("Bucket size", options=["1min", "5min", "15min", "1H", "1D"], value="5min")
            bucketed = (
                ts_df.set_index("timestamp")
                .groupby([pd.Grouper(freq=freq), "sentiment_label"])
                .size()
                .reset_index(name="count")
            )
            fig_tl = px.line(
                bucketed, x="timestamp", y="count", color="sentiment_label",
                color_discrete_map=SENTIMENT_COLORS, title="Sentiment Volume Over Time",
            )
            fig_tl.update_layout(
                template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                height=420, margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_tl, width='stretch')
            figs_for_report["Sentiment Over Time"] = fig_tl

            avg_over_time = ts_df.set_index("timestamp")["sentiment_score"].resample(freq).mean().reset_index()
            fig_avg = px.line(avg_over_time, x="timestamp", y="sentiment_score", title="Average Sentiment Score Over Time")
            fig_avg.add_hline(y=0, line_dash="dot", line_color="#555")
            fig_avg.update_traces(line_color="#5c9eff")
            fig_avg.update_layout(
                template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                height=380, margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_avg, width='stretch')

    # ── Topic Analysis ───────────────────────────────────────────────────────
    with tabs[3]:
        render_topic_analysis_tab(df, figs_for_report)

    # ── Segments ─────────────────────────────────────────────────────────────
    with tabs[4]:
        render_segments_tab(df, video_id)

    # ── Authors & Engagement ────────────────────────────────────────────────
    with tabs[5]:
        col1, col2 = st.columns(2)
        with col1:
            top_authors = df["author"].value_counts().head(15).reset_index()
            top_authors.columns = ["author", "messages"]
            fig_auth = px.bar(top_authors.sort_values("messages"), x="messages", y="author", orientation="h",
                               title="Most Active Authors")
            fig_auth.update_traces(marker_color="#b06fe0")
            fig_auth.update_layout(
                template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                height=460, margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_auth, width='stretch')

        with col2:
            if "badge" in df.columns:
                badge_counts = df["badge"].value_counts().reset_index()
                badge_counts.columns = ["badge", "count"]
                fig_badge = px.bar(badge_counts, x="badge", y="count", title="Messages by Author Type")
                fig_badge.update_traces(marker_color="#2fbf71")
                fig_badge.update_layout(
                    template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                    height=460, margin=dict(t=50, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_badge, width='stretch')

        st.markdown("#### Sentiment by author type")
        if "badge" in df.columns:
            by_badge = df.groupby(["badge", "sentiment_label"]).size().reset_index(name="count")
            fig_bb = px.bar(by_badge, x="badge", y="count", color="sentiment_label",
                             color_discrete_map=SENTIMENT_COLORS, barmode="group")
            fig_bb.update_layout(
                template="plotly_dark", paper_bgcolor="#080808", plot_bgcolor="#080808",
                height=380, margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bb, width='stretch')

    # ── Message Explorer ────────────────────────────────────────────────────
    with tabs[6]:
        colf1, colf2, colf3 = st.columns(3)
        f_source = colf1.multiselect("Source", sorted(df["source"].unique()), default=list(df["source"].unique()))
        f_sentiment = colf2.multiselect("Sentiment", ["Positive", "Neutral", "Negative"],
                                         default=["Positive", "Neutral", "Negative"])
        f_search = colf3.text_input("Search text contains...")

        filtered = df[df["source"].isin(f_source) & df["sentiment_label"].isin(f_sentiment)]
        if f_search:
            filtered = filtered[filtered["text"].str.contains(re.escape(f_search), case=False, na=False)]

        st.caption(f"{len(filtered):,} messages match your filters")
        st.dataframe(
            filtered[["source", "author", "text", "sentiment_label", "sentiment_score"]]
            .sort_values("sentiment_score", ascending=False),
            width='stretch', height=420,
        )

        st.markdown("#### Most Positive")
        for r in df.sort_values("sentiment_score", ascending=False).head(5).itertuples():
            st.markdown(
                f'<div class="quote-card quote-pos"><div class="quote-meta">{r.source} · {r.author} · score {r.sentiment_score:+.2f}</div>'
                f'<div class="quote-text">{r.text}</div></div>', unsafe_allow_html=True,
            )
        st.markdown("#### Most Negative")
        for r in df.sort_values("sentiment_score", ascending=True).head(5).itertuples():
            st.markdown(
                f'<div class="quote-card quote-neg"><div class="quote-meta">{r.source} · {r.author} · score {r.sentiment_score:+.2f}</div>'
                f'<div class="quote-text">{r.text}</div></div>', unsafe_allow_html=True,
            )

    # ── AI Insights ─────────────────────────────────────────────────────────
    with tabs[7]:
        if not run_ai_summary:
            st.info("AI theme summary was disabled for this run (toggle it in the sidebar and re-run).")
        elif st.session_state.get("ai_summary"):
            st.markdown(st.session_state["ai_summary"])
        else:
            if st.button("Generate AI theme summary now"):
                status = st.empty()
                def cb(msg):
                    status.info(msg)
                with st.spinner("Talking to Claude via Bedrock..."):
                    try:
                        summary = generate_ai_summary(df, video_id, scope, status_cb=cb)
                        st.session_state["ai_summary"] = summary
                        status.empty()
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

    # ── Chat with your data ─────────────────────────────────────────────────
    with tabs[8]:
        st.caption(
            f"Ask questions about the {len(df):,} analyzed messages/comments for `{video_id}`. "
            "Claude can search and aggregate the full dataset (not just a sample) via tool calls."
        )

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        if "chat_display" not in st.session_state:
            st.session_state["chat_display"] = []  # [(role, text)] for rendering only

        if not st.session_state["chat_display"]:
            st.markdown("**Try asking:**")
            suggestions = [
                "What are people most excited about?",
                "Show me the funniest negative comments",
                "How did sentiment change over time?",
                "Who are the most active positive commenters?",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                if cols[i % 2].button(sug, key=f"chat_suggestion_{i}", width='stretch'):
                    st.session_state["_pending_chat_input"] = sug

        for role, text in st.session_state["chat_display"]:
            with st.chat_message(role):
                st.markdown(text)

        pending = st.session_state.pop("_pending_chat_input", None)
        user_input = st.chat_input("Ask about this chat/comment data...")
        question = pending or user_input

        if question:
            st.session_state["chat_display"].append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = chat_with_data(df, video_id, scope, question)
                    except Exception as e:
                        answer = f"Something went wrong talking to Claude: {e}"
                st.markdown(answer)
            st.session_state["chat_display"].append(("assistant", answer))

        if st.session_state["chat_display"]:
            if st.button("Clear chat"):
                st.session_state["chat_messages"] = []
                st.session_state["chat_display"] = []
                st.rerun()

    # ── Export / Report ─────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown("#### Downloadable data")
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇ Download raw data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"sentiment_data_{video_id}.csv",
            mime="text/csv",
            width='stretch',
        )
        c2.download_button(
            "⬇ Download raw data (JSON)",
            df.to_json(orient="records", date_format="iso").encode("utf-8"),
            file_name=f"sentiment_data_{video_id}.json",
            mime="application/json",
            width='stretch',
        )

        st.divider()
        st.markdown("#### PDF report")
        st.caption("Bundles the KPIs, key charts, notable quotes, and the AI summary (if generated) into a shareable PDF.")
        if st.button("📄 Generate PDF report", width='stretch'):
            with st.spinner("Building report..."):
                ai_text = st.session_state.get("ai_summary") or "AI theme summary was not generated for this run."
                meta = {"video_id": video_id, "scope": scope}
                try:
                    pdf_bytes = build_pdf_report(meta, df, ai_text, figs_for_report)
                    st.session_state["pdf_bytes"] = pdf_bytes
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                "⬇ Download PDF report",
                st.session_state["pdf_bytes"],
                file_name=f"sentiment_report_{video_id}.pdf",
                mime="application/pdf",
                width='stretch',
            )

render_dashboard()
