
import re
import ast
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Streaming Marketing Intelligence",
    page_icon="🎬",
    layout="wide"
)

CARD_BG = "#1E293B"
FIG_BG = "#0F172A"
TEXT = "white"
MUTED = "#94A3B8"
ACCENT_1 = "#6366F1"
ACCENT_2 = "#10B981"
ACCENT_3 = "#F59E0B"
ACCENT_4 = "#EF4444"
ACCENT_5 = "#22C55E"

DEFAULT_DATASET_CANDIDATES = [
    
    "DATA/PROCESSED/all_streaming_titles.csv",
]

TOPIC_KEYWORDS = {
    "LGBTQ+": ["gay", "lesbian", "lgbt", "trans", "queer", "bisexual", "drag"],
    "Politics": ["politics", "president", "government", "election", "policy", "minister", "senate", "congress"],
    "Climate": ["climate", "environment", "global warming", "pollution", "sustainability", "ecology"],
    "War": ["war", "battle", "army", "soldier", "military", "conflict", "invasion"],
    "Family": ["family", "mother", "father", "children", "home", "parent", "siblings"],
    "Crime": ["crime", "murder", "police", "detective", "investigation", "gang", "mafia", "robbery"],
    "Romance": ["love", "romance", "relationship", "couple", "marriage", "affair"],
    "Technology": ["technology", "ai", "robot", "future", "cyber", "machine", "algorithm"],
    "Mental Health": ["depression", "anxiety", "trauma", "therapy", "mental", "psychological"],
    "Coming of Age": ["teen", "adolescent", "growing up", "school", "youth", "friendship"],
    "Social Issues": ["racism", "inequality", "poverty", "discrimination", "justice", "migration"],
    "Fantasy / Supernatural": ["magic", "witch", "dragon", "supernatural", "monster", "curse", "fantasy"],
}


# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    padding: 14px 16px;
    border-radius: 16px;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
}
.small-card {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
}
.objective-card {
    background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
    border: 1px solid #1E293B;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}
.muted {
    color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def metric_str(value, decimals=2):
    return f"{value:.{decimals}f}" if pd.notna(value) else "N/A"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_genres(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [g.strip() for g in text.split(",") if g.strip()]


def detect_topics_from_synopsis(text: str):
    text_clean = clean_text(text)
    found_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in text_clean for keyword in keywords):
            found_topics.append(topic)

    if not found_topics:
        found_topics.append("General / No clear topic")

    return found_topics


def get_active_topics_from_row(row):
    topics = []

    if "top_topics" in row.index and pd.notna(row.get("top_topics")) and str(row.get("top_topics")).strip():
        raw = str(row.get("top_topics"))
        if "|" in raw:
            topics.extend([x.strip() for x in raw.split("|") if x.strip()])
        elif "," in raw:
            topics.extend([x.strip() for x in raw.split(",") if x.strip()])
        else:
            topics.append(raw.strip())

    topic_like_cols = [c for c in row.index if c.startswith("topic_") and c != "topic_diversity_score"]
    for col in topic_like_cols:
        try:
            if float(row.get(col, 0)) == 1:
                topics.append(col.replace("topic_", "").replace("_", " ").title())
        except Exception:
            continue

    topics = [t for t in topics if t]
    return sorted(list(set(topics)))


def make_dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT)
    return fig, ax


def normalize_0_100(series):
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    min_v = valid.min()
    max_v = valid.max()
    if min_v == max_v:
        return pd.Series(50.0, index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 100


def ensure_business_value_score(df):
    if "business_value_score" in df.columns and df["business_value_score"].notna().any():
        return df

    pop_norm = normalize_0_100(df["popularity"]) if "popularity" in df.columns else pd.Series(np.nan, index=df.index)
    vote_avg_norm = normalize_0_100(df["vote_average"]) if "vote_average" in df.columns else pd.Series(np.nan, index=df.index)
    vote_count_norm = normalize_0_100(df["vote_count"]) if "vote_count" in df.columns else pd.Series(np.nan, index=df.index)
    visibility = normalize_0_100(df["visibility_score"]) if "visibility_score" in df.columns else pd.Series(np.nan, index=df.index)
    engagement = normalize_0_100(df["engagement_score"]) if "engagement_score" in df.columns else pd.Series(np.nan, index=df.index)
    reception = normalize_0_100(df["audience_reception_score"]) if "audience_reception_score" in df.columns else pd.Series(np.nan, index=df.index)

    pieces = pd.concat(
        [pop_norm, vote_avg_norm, vote_count_norm, visibility, engagement, reception],
        axis=1
    )
    pieces.columns = ["pop", "vote_avg", "vote_count", "visibility", "engagement", "reception"]

    weights = {
        "pop": 0.25,
        "vote_avg": 0.20,
        "vote_count": 0.15,
        "visibility": 0.15,
        "engagement": 0.10,
        "reception": 0.15,
    }

    weighted = sum(pieces[col].fillna(pieces.mean(axis=1)) * w for col, w in weights.items())
    df["business_value_score"] = weighted.round(2)

    return df


def estimate_synopsis_business_value(similar_df):
    if similar_df.empty:
        return np.nan

    score_col = None
    if "predicted_business_value" in similar_df.columns and similar_df["predicted_business_value"].notna().any():
        score_col = "predicted_business_value"
    elif "business_value_score" in similar_df.columns and similar_df["business_value_score"].notna().any():
        score_col = "business_value_score"

    if score_col is None:
        return np.nan

    work = similar_df[[score_col, "similarity_score"]].copy()
    work = work.dropna()

    if work.empty:
        return np.nan

    weights = work["similarity_score"].clip(lower=0.001)
    value = np.average(work[score_col], weights=weights)
    return round(float(value), 2)


def business_value_band(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 75:
        return "High commercial potential"
    if score >= 55:
        return "Medium-high potential"
    if score >= 40:
        return "Moderate potential"
    return "Lower potential"


def dataset_health_text(df):
    parts = []
    parts.append(f"{len(df):,} titles")
    if "content_type" in df.columns:
        parts.append(f"{df['content_type'].nunique()} content types")
    if "source" in df.columns:
        parts.append(f"{df['source'].nunique()} sources")
    if "release_year" in df.columns and df["release_year"].notna().any():
        parts.append(
            f"years {int(df['release_year'].min())}–{int(df['release_year'].max())}"
        )
    return " · ".join(parts)


def choose_dataset_file():
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    dataset_path = choose_dataset_file()
    if dataset_path is None:
        raise FileNotFoundError(
            "No dataset found. Please place one of these files in the app folder: "
            "final_streaming_dataset.csv, streamlit_ready_dataset.csv, all_streaming_titles.csv"
        )

    df = pd.read_csv(dataset_path)

    # Core cleanup
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    numeric_cols = [
        "popularity", "vote_average", "vote_count",
        "visibility_score", "engagement_score",
        "business_value_score", "predicted_business_value",
        "topic_diversity_score", "runtime_final", "cluster",
        "audience_reception_score", "freshness_score",
        "release_year"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "title", "overview", "overview_clean", "genre_names",
        "content_type", "source", "original_language",
        "marketing_segment", "cluster_label", "production_companies",
        "network", "status", "web_channel", "top_topics"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "genre_names" in df.columns:
        df["genre_list"] = df["genre_names"].apply(parse_genres)
    else:
        df["genre_list"] = [[] for _ in range(len(df))]

    if "overview_clean" not in df.columns:
        df["overview_clean"] = df["overview"].fillna("").apply(clean_text)
    else:
        df["overview_clean"] = df["overview_clean"].fillna("").apply(clean_text)

    if "title" not in df.columns:
        df["title"] = "Untitled"

    if "content_type" not in df.columns:
        df["content_type"] = "unknown"

    df["title_key"] = df["title"].fillna("").astype(str).str.lower().str.strip()
    df["detected_topics_auto"] = df["overview"].fillna("").apply(detect_topics_from_synopsis)
    df["detected_topics_text"] = df["detected_topics_auto"].apply(lambda x: ", ".join(x) if x else "")

    df = ensure_business_value_score(df)

    if "predicted_business_value" not in df.columns:
        df["predicted_business_value"] = np.nan

    if "cluster_label" not in df.columns:
        df["cluster_label"] = ""

    if "marketing_segment" not in df.columns:
        df["marketing_segment"] = ""

    return df, dataset_path


@st.cache_data
def prepare_similarity(df):
    work_df = df.copy()

    work_df["topics_text"] = work_df.apply(
        lambda row: " ".join(get_active_topics_from_row(row)) if get_active_topics_from_row(row) else row.get("detected_topics_text", ""),
        axis=1
    )

    genre_text = work_df["genre_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    work_df["similarity_text"] = (
        work_df["title"].fillna("") + " "
        + genre_text.fillna("") + " "
        + work_df["overview_clean"].fillna("") + " "
        + work_df["topics_text"].fillna("") + " "
        + work_df["original_language"].fillna("")
    ).str.lower()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(work_df["similarity_text"])

    return work_df, matrix


def get_similar_titles(df_similarity, matrix, selected_title, top_n=10):
    matches = df_similarity.index[df_similarity["title"] == selected_title].tolist()
    if not matches:
        return pd.DataFrame()

    idx = matches[0]
    sims = cosine_similarity(matrix[idx], matrix).flatten()

    sim_df = df_similarity.copy()
    sim_df["similarity_score"] = sims
    sim_df = sim_df[sim_df.index != idx].copy()

    sort_cols = ["similarity_score", "business_value_score", "vote_average", "popularity"]
    sort_cols = [c for c in sort_cols if c in sim_df.columns]

    rank_cols = [
        "title", "content_type", "genre_names", "release_year",
        "vote_average", "popularity", "visibility_score",
        "engagement_score", "business_value_score",
        "predicted_business_value", "marketing_segment",
        "cluster_label", "detected_topics_text", "similarity_score", "overview"
    ]
    existing_cols = [c for c in rank_cols if c in sim_df.columns]

    return sim_df.sort_values(
        by=sort_cols,
        ascending=[False] * len(sort_cols)
    )[existing_cols].head(top_n)


def get_top_similar_titles_from_synopsis(user_synopsis, content_type, df_similarity, matrix, top_n=5):
    working = df_similarity.copy()

    if content_type and content_type.lower() != "all" and "content_type" in working.columns:
        working = working[
            working["content_type"].astype(str).str.lower() == content_type.lower()
        ].copy()

    if working.empty:
        return pd.DataFrame()

    user_text = clean_text(user_synopsis)
    user_topics = set(detect_topics_from_synopsis(user_synopsis))

    user_vec = matrix.__class__(matrix.shape[0])  # placeholder not used directly

    # Need to rebuild on filtered subset from existing vectorizer behavior:
    # simplest safe path: refit a local vectorizer on filtered rows
    local_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2)
    )
    local_matrix = local_vectorizer.fit_transform(working["similarity_text"])
    user_vec = local_vectorizer.transform([user_text])
    text_sims = cosine_similarity(user_vec, local_matrix).flatten()

    working["text_similarity"] = text_sims

    def topic_overlap_score(item_topics_text):
        item_topics = set([x.strip() for x in str(item_topics_text).split(",") if x.strip()])
        if not user_topics or not item_topics:
            return 0
        return len(user_topics.intersection(item_topics)) / max(len(user_topics), 1)

    working["topic_similarity"] = working["detected_topics_text"].apply(topic_overlap_score)

    working["similarity_score"] = (
        0.80 * working["text_similarity"] +
        0.20 * working["topic_similarity"]
    )

    cols_to_show = [
        col for col in [
            "title",
            "content_type",
            "genre_names",
            "original_language",
            "release_year",
            "overview",
            "business_value_score",
            "predicted_business_value",
            "marketing_segment",
            "cluster_label",
            "detected_topics_text",
            "text_similarity",
            "topic_similarity",
            "similarity_score"
        ] if col in working.columns
    ]

    result = working.sort_values(
        by=["similarity_score", "business_value_score"],
        ascending=[False, False]
    ).head(top_n)

    return result[cols_to_show]


def get_rank(df_all, title, col):
    if col not in df_all.columns:
        return None, None

    rank_df = df_all[["title", col]].dropna().sort_values(by=col, ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    matches = rank_df[rank_df["title"] == title]

    if matches.empty:
        return None, len(rank_df)

    return int(matches.index[0]), len(rank_df)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────────────────────
df, dataset_path = load_data()
df_similarity, sim_matrix = prepare_similarity(df)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🎬 Streaming Marketing Intelligence Dashboard")

st.markdown(f"""
<div class="objective-card">
    <div class="section-title">Marketing Objective</div>
    <div class="muted">
        Support content marketing and acquisition decisions by identifying which titles show stronger commercial potential,
        what themes are most attractive, and which existing titles can serve as relevant comparables for a new idea or synopsis.
    </div>
    <br>
    <div class="muted">
        <strong>Primary use cases:</strong> content benchmarking · similarity search · idea validation · business value estimation · marketing storytelling
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"Dataset loaded: `{dataset_path}` · {dataset_health_text(df)}")
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

content_options_real = sorted([x for x in df["content_type"].dropna().astype(str).unique().tolist() if x != ""])
content_options = ["All"] + content_options_real
selected_content = st.sidebar.selectbox("Content type", options=content_options, index=0)

source_options_real = sorted([x for x in df["source"].dropna().astype(str).unique().tolist() if x != ""]) if "source" in df.columns else []
source_options = ["All"] + source_options_real
selected_source = st.sidebar.selectbox("Source", options=source_options, index=0)

language_options_real = sorted([x for x in df["original_language"].dropna().astype(str).unique().tolist() if x != ""]) if "original_language" in df.columns else []
selected_languages = st.sidebar.multiselect(
    "Languages",
    options=["All"] + language_options_real,
    default=["All"]
)

all_detected_topics = sorted({
    topic
    for topics in df["detected_topics_auto"]
    for topic in topics
})
selected_topics = st.sidebar.multiselect(
    "Detected themes",
    options=all_detected_topics,
    default=[]
)

if "release_year" in df.columns and df["release_year"].notna().any():
    min_year = int(df["release_year"].dropna().min())
    max_year = int(df["release_year"].dropna().max())
    year_range = st.sidebar.slider(
        "Release year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_range = None

st.sidebar.divider()
st.sidebar.markdown("### Quick title search")

sidebar_title_options = sorted(df["title"].dropna().astype(str).unique().tolist())
sidebar_selected_title = st.sidebar.selectbox(
    "Search a title",
    options=sidebar_title_options,
    index=None,
    placeholder="Start typing..."
)


# ──────────────────────────────────────────────────────────────────────────────
# FILTERS
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df.copy()

if selected_content != "All" and "content_type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["content_type"] == selected_content]

if selected_source != "All" and "source" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["source"] == selected_source]

if "All" not in selected_languages and selected_languages and "original_language" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["original_language"].isin(selected_languages)]

if year_range and "release_year" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["release_year"].between(year_range[0], year_range[1], inclusive="both")
    ]

if selected_topics:
    filtered_df = filtered_df[
        filtered_df["detected_topics_auto"].apply(
            lambda x: any(topic in x for topic in selected_topics)
        )
    ]


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR QUICK RESULT
# ──────────────────────────────────────────────────────────────────────────────
if sidebar_selected_title:
    sidebar_row = df[df["title"] == sidebar_selected_title].head(1)
    if not sidebar_row.empty:
        row = sidebar_row.iloc[0]

        st.sidebar.markdown("#### Title snapshot")
        st.sidebar.markdown(
            f"**Type:** {row.get('content_type', 'N/A')}  \n"
            f"**Year:** {int(row['release_year']) if pd.notna(row.get('release_year')) else 'N/A'}  \n"
            f"**Language:** {row.get('original_language', 'N/A')}"
        )

        st.sidebar.markdown("#### Overview")
        overview_text = row.get("overview", "")
        if overview_text:
            st.sidebar.caption(overview_text[:350] + ("..." if len(overview_text) > 350 else ""))
        else:
            st.sidebar.caption("No overview available.")

        st.sidebar.markdown("#### Performance")
        st.sidebar.metric("Visibility", metric_str(row.get("visibility_score", np.nan)))
        st.sidebar.metric("Engagement", metric_str(row.get("engagement_score", np.nan)))
        st.sidebar.metric("Business Value", metric_str(row.get("business_value_score", np.nan)))
        if pd.notna(row.get("predicted_business_value", np.nan)):
            st.sidebar.metric("Predicted BV", metric_str(row.get("predicted_business_value", np.nan)))

        active_topics = get_active_topics_from_row(row)
        if not active_topics:
            active_topics = row.get("detected_topics_auto", [])
        st.sidebar.markdown("#### Themes")
        st.sidebar.caption(", ".join(active_topics) if active_topics else "Not detected")


# ──────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"### Filtered catalog: {filtered_df.shape[0]:,} titles")

total_titles = len(filtered_df)
avg_popularity = filtered_df["popularity"].mean() if "popularity" in filtered_df.columns else np.nan
avg_vote = filtered_df["vote_average"].mean() if "vote_average" in filtered_df.columns else np.nan
avg_business = filtered_df["business_value_score"].mean() if "business_value_score" in filtered_df.columns else np.nan
avg_visibility = filtered_df["visibility_score"].mean() if "visibility_score" in filtered_df.columns else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total titles", f"{total_titles:,}")
k2.metric("Avg popularity", metric_str(avg_popularity))
k3.metric("Avg rating", metric_str(avg_vote))
k4.metric("Avg business value", metric_str(avg_business))

k5, k6, k7, k8 = st.columns(4)
k5.metric("Avg visibility", metric_str(avg_visibility))
k6.metric("Movies", f"{int((filtered_df['content_type'].astype(str).str.lower() == 'movie').sum()):,}" if "content_type" in filtered_df.columns else "N/A")
k7.metric("Series / TV", f"{int((filtered_df['content_type'].astype(str).str.lower().isin(['tv', 'series'])).sum()):,}" if "content_type" in filtered_df.columns else "N/A")
k8.metric("Sources", f"{filtered_df['source'].nunique():,}" if "source" in filtered_df.columns else "N/A")

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview",
    "🎯 Marketing Performance",
    "🧠 Audience & Themes",
    "🔎 Similar Titles Finder",
    "📝 Synopsis Opportunity Lab",
    "🗂 Dataset Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Executive overview")

    c1, c2 = st.columns(2)

    with c1:
        if "content_type" in filtered_df.columns and not filtered_df.empty:
            counts = filtered_df["content_type"].value_counts()
            fig, ax = make_dark_fig((7, 4))
            ax.bar(counts.index, counts.values, color=ACCENT_1)
            ax.set_title("Titles by content type", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("Count", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "source" in filtered_df.columns and not filtered_df.empty:
            counts = filtered_df["source"].value_counts().head(10)
            fig, ax = make_dark_fig((7, 4))
            ax.bar(counts.index, counts.values, color=ACCENT_2)
            ax.set_title("Top sources", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("Count", color=TEXT)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)
            plt.close(fig)

    c3, c4 = st.columns(2)

    with c3:
        if "popularity" in filtered_df.columns and filtered_df["popularity"].notna().any():
            fig, ax = make_dark_fig((7, 4))
            ax.hist(filtered_df["popularity"].dropna(), bins=30, color=ACCENT_3)
            ax.set_title("Popularity distribution", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Popularity", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c4:
        if "business_value_score" in filtered_df.columns and filtered_df["business_value_score"].notna().any():
            fig, ax = make_dark_fig((7, 4))
            ax.hist(filtered_df["business_value_score"].dropna(), bins=30, color=ACCENT_5)
            ax.set_title("Business value distribution", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Business Value Score", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    if "release_year" in filtered_df.columns and filtered_df["release_year"].notna().any():
        yearly = filtered_df["release_year"].dropna().astype(int).value_counts().sort_index()
        fig, ax = make_dark_fig((12, 4))
        ax.plot(yearly.index, yearly.values)
        ax.set_title("Catalog trend by release year", color=TEXT, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Release year", color=TEXT)
        ax.set_ylabel("Titles", color=TEXT)
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MARKETING PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Marketing performance analysis")

    metric_map = {
        "Visibility": "visibility_score",
        "Engagement": "engagement_score",
        "Business Value": "business_value_score",
        "Popularity": "popularity",
        "Audience Reception": "audience_reception_score"
    }

    available_metric_labels = [k for k, v in metric_map.items() if v in filtered_df.columns]
    selected_metric_label = st.radio(
        "Highlighted KPI",
        options=available_metric_labels,
        horizontal=True,
        index=available_metric_labels.index("Business Value") if "Business Value" in available_metric_labels else 0
    )
    highlight_metric = metric_map[selected_metric_label]

    c1, c2 = st.columns(2)

    with c1:
        top_df = filtered_df[["title", highlight_metric]].dropna().sort_values(
            by=highlight_metric, ascending=False
        ).head(10)

        if not top_df.empty:
            fig, ax = make_dark_fig((8, 5))
            ax.barh(top_df["title"][::-1], top_df[highlight_metric][::-1], color=ACCENT_1)
            ax.set_title(f"Top 10 by {selected_metric_label}", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Score", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "marketing_segment" in filtered_df.columns and filtered_df["marketing_segment"].astype(str).str.strip().ne("").any():
            seg = filtered_df["marketing_segment"].replace("", "Unknown").value_counts().head(10)
            fig, ax = make_dark_fig((8, 5))
            ax.bar(seg.index, seg.values, color=ACCENT_2)
            ax.set_title("Marketing segment distribution", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No `marketing_segment` column available in the current dataset.")

    if all(c in filtered_df.columns for c in ["popularity", "business_value_score"]):
        sample_df = filtered_df[["popularity", "business_value_score"]].dropna().sample(
            min(5000, len(filtered_df[["popularity", "business_value_score"]].dropna())),
            random_state=42
        )
        fig, ax = make_dark_fig((10, 5))
        ax.scatter(sample_df["popularity"], sample_df["business_value_score"], alpha=0.35)
        ax.set_title("Popularity vs Business Value", color=TEXT, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Popularity", color=TEXT)
        ax.set_ylabel("Business Value Score", color=TEXT)
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – AUDIENCE & THEMES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Audience and thematic signals")

    c1, c2 = st.columns(2)

    with c1:
        topic_counts = {}
        for topic_list in filtered_df["detected_topics_auto"]:
            for topic in topic_list:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if topic_counts:
            topic_series = pd.Series(topic_counts).sort_values(ascending=False).head(10)
            fig, ax = make_dark_fig((8, 5))
            ax.barh(topic_series.index[::-1], topic_series.values[::-1], color=ACCENT_3)
            ax.set_title("Top detected themes", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Count", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "original_language" in filtered_df.columns:
            lang_counts = filtered_df["original_language"].replace("", "Unknown").value_counts().head(10)
            fig, ax = make_dark_fig((8, 5))
            ax.bar(lang_counts.index, lang_counts.values, color=ACCENT_4)
            ax.set_title("Top languages", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)
            plt.close(fig)

    if "genre_names" in filtered_df.columns:
        exploded = (
            filtered_df.assign(genre_item=filtered_df["genre_list"])
            .explode("genre_item")
        )
        exploded = exploded[exploded["genre_item"].notna() & (exploded["genre_item"] != "")]
        if not exploded.empty:
            top_genres = exploded["genre_item"].value_counts().head(12)
            fig, ax = make_dark_fig((10, 5))
            ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=ACCENT_5)
            ax.set_title("Top genres", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Count", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – SIMILAR TITLES FINDER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Similar titles finder")

    available_titles = sorted(filtered_df["title"].dropna().astype(str).unique().tolist())

    if not available_titles:
        st.warning("No titles available under the current filters.")
    else:
        selected_title = st.selectbox(
            "Start typing and select a title",
            options=available_titles,
            index=None,
            placeholder="Type a movie or series title..."
        )

        similar_df = pd.DataFrame()

        if selected_title:
            selected_row = filtered_df[filtered_df["title"] == selected_title].head(1)

            if not selected_row.empty:
                row = selected_row.iloc[0]

                rank_pop, total_pop = get_rank(df, selected_title, "popularity")
                rank_vote, total_vote = get_rank(df, selected_title, "vote_average")
                rank_business, total_business = get_rank(df, selected_title, "business_value_score")

                st.markdown(f"#### {selected_title}")

                info1, info2, info3, info4 = st.columns(4)
                info1.metric("Type", row.get("content_type", "N/A"))
                info2.metric("Year", int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A")
                info3.metric("Rating", metric_str(row.get("vote_average", np.nan)))
                info4.metric("Business Value", metric_str(row.get("business_value_score", np.nan)))

                st.markdown(
                    f"**Genres:** {row.get('genre_names', 'N/A')}  \n"
                    f"**Language:** {row.get('original_language', 'N/A')}  \n"
                    f"**Marketing Segment:** {row.get('marketing_segment', 'N/A')}  \n"
                    f"**Cluster:** {row.get('cluster_label', 'N/A') if str(row.get('cluster_label', '')).strip() else 'N/A'}"
                )

                active_topics = get_active_topics_from_row(row)
                if not active_topics:
                    active_topics = row.get("detected_topics_auto", [])
                st.markdown(f"**Detected themes:** {', '.join(active_topics) if active_topics else 'Not detected'}")

                if row.get("overview"):
                    st.markdown("**Overview**")
                    st.write(row["overview"])

                r1, r2, r3 = st.columns(3)
                r1.metric("Popularity rank", f"#{rank_pop}" if rank_pop else "N/A", delta=f"of {total_pop}" if total_pop else None)
                r2.metric("Rating rank", f"#{rank_vote}" if rank_vote else "N/A", delta=f"of {total_vote}" if total_vote else None)
                r3.metric("Business rank", f"#{rank_business}" if rank_business else "N/A", delta=f"of {total_business}" if total_business else None)

                st.divider()

                similar_df = get_similar_titles(df_similarity, sim_matrix, selected_title, top_n=10)

                st.markdown("#### Most similar titles")

                if similar_df.empty:
                    st.info("No similar titles found.")
                else:
                    fig, ax = make_dark_fig((9, 5))
                    plot_df = similar_df.head(8).copy()
                    ax.barh(plot_df["title"][::-1], plot_df["similarity_score"][::-1], color=ACCENT_1)
                    ax.set_title("Top similarity", color=TEXT, fontsize=13, fontweight="bold", pad=10)
                    ax.set_xlabel("Similarity Score", color=TEXT)
                    st.pyplot(fig)
                    plt.close(fig)

                    display_df = similar_df.copy()
                    if "similarity_score" in display_df.columns:
                        display_df["similarity_score"] = display_df["similarity_score"].round(3)

                    st.dataframe(display_df, use_container_width=True)

        if not similar_df.empty:
            csv_similar = similar_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download similar titles",
                data=csv_similar,
                file_name="similar_titles_results.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – SYNOPSIS OPPORTUNITY LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Synopsis opportunity lab")
    st.markdown(
        """
        Paste a new synopsis and the app will:
        1. detect its likely themes,
        2. retrieve the most similar titles in the dataset,
        3. estimate a likely business value based on comparable content.
        """
    )

    user_type = st.selectbox(
        "Content type",
        options=["All", "movie", "tv"],
        index=0
    )

    user_synopsis = st.text_area(
        "Paste the synopsis",
        height=180,
        placeholder="Example: A young journalist uncovers a political corruption network while trying to protect her family and save her career..."
    )

    top_n = st.slider("Number of comparables", min_value=3, max_value=10, value=5)

    if st.button("Analyze synopsis"):
        if not user_synopsis or not user_synopsis.strip():
            st.warning("Please paste a synopsis first.")
        else:
            detected_topics = detect_topics_from_synopsis(user_synopsis)

            synopsis_results = get_top_similar_titles_from_synopsis(
                user_synopsis=user_synopsis,
                content_type=user_type,
                df_similarity=df_similarity,
                matrix=sim_matrix,
                top_n=top_n
            )

            estimated_bv = estimate_synopsis_business_value(synopsis_results)
            estimated_band = business_value_band(estimated_bv)

            st.session_state["synopsis_results"] = synopsis_results
            st.session_state["user_synopsis_text"] = user_synopsis
            st.session_state["user_synopsis_topics"] = detected_topics
            st.session_state["user_synopsis_type"] = user_type
            st.session_state["estimated_bv"] = estimated_bv
            st.session_state["estimated_band"] = estimated_band

    if "synopsis_results" in st.session_state:
        synopsis_results = st.session_state["synopsis_results"]
        detected_topics = st.session_state.get("user_synopsis_topics", [])
        user_synopsis_saved = st.session_state.get("user_synopsis_text", "")
        user_type_saved = st.session_state.get("user_synopsis_type", "All")
        estimated_bv = st.session_state.get("estimated_bv", np.nan)
        estimated_band = st.session_state.get("estimated_band", "Unknown")

        a1, a2, a3 = st.columns(3)
        a1.metric("Detected themes", f"{len(detected_topics)}")
        a2.metric("Estimated Business Value", metric_str(estimated_bv))
        a3.metric("Commercial assessment", estimated_band)

        st.markdown("### Detected themes")
        st.write(", ".join(detected_topics) if detected_topics else "No clear themes detected.")

        st.markdown("### Similar titles")
        if synopsis_results.empty:
            st.info("No comparable titles were found for the current filters.")
        else:
            fig, ax = make_dark_fig((9, 5))
            plot_df = synopsis_results.head(8).copy()
            ax.barh(plot_df["title"][::-1], plot_df["similarity_score"][::-1], color=ACCENT_2)
            ax.set_title("Top comparable titles by similarity", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Similarity Score", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

            for i, (_, row) in enumerate(synopsis_results.iterrows(), start=1):
                st.markdown(f"#### #{i} - {row.get('title', 'Unknown title')}")
                st.write(f"**Type:** {row.get('content_type', 'N/A')}")
                st.write(f"**Genres:** {row.get('genre_names', 'N/A')}")
                st.write(f"**Language:** {row.get('original_language', 'N/A')}")
                st.write(f"**Year:** {row.get('release_year', 'N/A')}")
                st.write(f"**Text similarity:** {row.get('text_similarity', 0):.3f}")
                st.write(f"**Topic similarity:** {row.get('topic_similarity', 0):.3f}")
                st.write(f"**Final similarity score:** {row.get('similarity_score', 0):.3f}")

                if "business_value_score" in row and pd.notna(row["business_value_score"]):
                    st.write(f"**Business Value Score:** {row['business_value_score']:.2f}")

                if "predicted_business_value" in row and pd.notna(row["predicted_business_value"]):
                    st.write(f"**Predicted Business Value:** {row['predicted_business_value']:.2f}")

                if "marketing_segment" in row and str(row.get("marketing_segment", "")).strip():
                    st.write(f"**Marketing Segment:** {row.get('marketing_segment')}")

                st.write(f"**Overview:** {row.get('overview', 'N/A')}")
                st.markdown("---")

            st.markdown("### Opportunity readout")
            if pd.notna(estimated_bv):
                if estimated_bv >= 75:
                    st.success(
                        "This synopsis resembles titles with strong commercial potential. "
                        "It may be a good candidate for premium positioning, stronger promotion, or greenlight discussion."
                    )
                elif estimated_bv >= 55:
                    st.info(
                        "This synopsis resembles titles with medium-high potential. "
                        "It could perform well depending on packaging, cast, timing, and campaign strategy."
                    )
                elif estimated_bv >= 40:
                    st.warning(
                        "This synopsis resembles titles with moderate potential. "
                        "It may require sharper positioning or a more differentiated marketing angle."
                    )
                else:
                    st.error(
                        "This synopsis resembles titles with lower estimated commercial value in the current dataset."
                    )
            else:
                st.info("Business value estimation could not be calculated from the available comparable titles.")

            export_df = synopsis_results.copy()
            export_df.insert(0, "input_synopsis", user_synopsis_saved)
            export_df.insert(1, "input_content_type", user_type_saved)
            export_df.insert(2, "detected_topics", ", ".join(detected_topics))
            export_df.insert(3, "estimated_business_value", estimated_bv)
            export_df.insert(4, "estimated_business_value_band", estimated_band)

            csv_synopsis = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download synopsis analysis",
                data=csv_synopsis,
                file_name="synopsis_opportunity_results.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Dataset explorer")

    explorer_cols = [c for c in [
        "title", "content_type", "genre_names", "release_year",
        "original_language", "source", "popularity", "vote_average",
        "vote_count", "visibility_score", "engagement_score",
        "audience_reception_score", "business_value_score", "predicted_business_value",
        "marketing_segment", "cluster_label", "detected_topics_text", "overview"
    ] if c in filtered_df.columns]

    st.dataframe(filtered_df[explorer_cols].head(200), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download filtered dataset",
        data=csv,
        file_name="filtered_streaming_dataset.csv",
        mime="text/csv"
    )