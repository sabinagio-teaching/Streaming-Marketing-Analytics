
import re
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
    page_title="Streaming Marketing Analytics",
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

TOPIC_COLS = [
    "topic_lgbtq",
    "topic_politics",
    "topic_climate",
    "topic_war",
    "topic_family",
    "topic_crime",
    "topic_romance",
    "topic_technology",
]

TOPIC_LABELS = {
    "topic_lgbtq": "LGBTQ+",
    "topic_politics": "Politics",
    "topic_climate": "Climate",
    "topic_war": "War",
    "topic_family": "Family",
    "topic_crime": "Crime",
    "topic_romance": "Romance",
    "topic_technology": "Technology",
}

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
    padding-top: 1.4rem;
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
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_topics_from_synopsis(text):
    text_clean = clean_text(text)
    found_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in text_clean for keyword in keywords):
            found_topics.append(topic)

    if not found_topics:
        found_topics.append("General / No clear topic")

    return found_topics


def get_active_topics_from_row(row):
    return [TOPIC_LABELS[col] for col in TOPIC_COLS if row.get(col, 0) == 1]


def make_dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT)
    return fig, ax


def safe_float(value):
    return float(value) if pd.notna(value) else np.nan


def metric_str(value, decimals=2):
    return f"{value:.{decimals}f}" if pd.notna(value) else "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("final_streaming_dataset.csv")

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    numeric_cols = [
        "popularity", "vote_average", "vote_count",
        "visibility_score", "engagement_score",
        "business_value_score", "predicted_business_value",
        "topic_diversity_score", "runtime_final", "cluster"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "title", "overview", "overview_clean", "genre_names",
        "content_type", "source", "original_language",
        "marketing_segment", "cluster_label", "production_companies"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in TOPIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    df["title_key"] = df["title"].fillna("").astype(str).str.lower().str.strip()

    if "overview_clean" not in df.columns:
        df["overview_clean"] = df["overview"].fillna("").apply(clean_text)
    else:
        df["overview_clean"] = df["overview_clean"].fillna("").apply(clean_text)

    return df


@st.cache_data
def prepare_similarity(df):
    work_df = df.copy()

    work_df["topics_text"] = work_df.apply(
        lambda row: " ".join(get_active_topics_from_row(row)),
        axis=1
    )

    work_df["similarity_text"] = (
        work_df["title"].fillna("") + " "
        + work_df["genre_names"].fillna("").str.replace(",", " ", regex=False) + " "
        + work_df["overview_clean"].fillna("") + " "
        + work_df["overview"].fillna("") + " "
        + work_df["topics_text"].fillna("")
    ).str.lower()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(work_df["similarity_text"])

    return work_df, matrix


@st.cache_resource
def build_similarity_model(df):
    model_df = df.copy()

    if "overview" not in model_df.columns:
        model_df["overview"] = ""

    model_df["overview_clean"] = model_df["overview"].fillna("").apply(clean_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(model_df["overview_clean"])

    return vectorizer, tfidf_matrix, model_df


def get_similar_titles(df_similarity, matrix, selected_title, top_n=10):
    matches = df_similarity.index[df_similarity["title"] == selected_title].tolist()
    if not matches:
        return pd.DataFrame()

    idx = matches[0]
    sims = cosine_similarity(matrix[idx], matrix).flatten()

    sim_df = df_similarity.copy()
    sim_df["similarity_score"] = sims
    sim_df = sim_df[sim_df.index != idx].copy()

    rank_cols = [
        "title", "content_type", "genre_names", "release_year",
        "vote_average", "popularity", "visibility_score",
        "engagement_score", "business_value_score",
        "predicted_business_value", "marketing_segment",
        "cluster_label", "similarity_score"
    ]
    existing_cols = [c for c in rank_cols if c in sim_df.columns]

    return sim_df.sort_values(
        by=["similarity_score", "business_value_score", "vote_average", "popularity"],
        ascending=[False, False, False, False]
    )[existing_cols].head(top_n)


def get_top_similar_titles(user_synopsis, content_type, df, top_n=3):
    vectorizer, tfidf_matrix, model_df = build_similarity_model(df)

    filtered = model_df.copy()

    if "content_type" in filtered.columns and content_type and content_type.lower() != "all":
        filtered = filtered[
            filtered["content_type"].astype(str).str.lower() == content_type.lower()
        ].copy()

    if filtered.empty:
        return pd.DataFrame()

    filtered_indices = filtered.index.tolist()

    user_text = clean_text(user_synopsis)
    user_vec = vectorizer.transform([user_text])

    similarities = cosine_similarity(user_vec, tfidf_matrix[filtered_indices]).flatten()
    filtered["text_similarity"] = similarities

    user_topics = set(detect_topics_from_synopsis(user_synopsis))

    if "overview" in filtered.columns:
        filtered["detected_topics_temp"] = filtered["overview"].fillna("").apply(detect_topics_from_synopsis)

        def topic_overlap_score(topics_list):
            item_topics = set(topics_list)
            if not user_topics or not item_topics:
                return 0
            return len(user_topics.intersection(item_topics)) / max(len(user_topics), 1)

        filtered["topic_similarity"] = filtered["detected_topics_temp"].apply(topic_overlap_score)
    else:
        filtered["topic_similarity"] = 0

    filtered["similarity_score"] = (
        0.8 * filtered["text_similarity"] +
        0.2 * filtered["topic_similarity"]
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
            "text_similarity",
            "topic_similarity",
            "similarity_score"
        ] if col in filtered.columns
    ]

    result = filtered.sort_values("similarity_score", ascending=False).head(top_n)
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
df = load_data()
df_similarity, sim_matrix = prepare_similarity(df)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎬 Streaming Marketing Analytics")
st.markdown("**Dashboard de visibilidad, engagement, valor comercial y similitud entre títulos**")
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

content_options_real = sorted([x for x in df["content_type"].dropna().unique().tolist() if x != ""])
content_options = ["All"] + content_options_real
selected_content = st.sidebar.selectbox(
    "Tipo de contenido",
    options=content_options,
    index=0
)

source_options_real = sorted([x for x in df["source"].dropna().unique().tolist() if x != ""])
source_options = ["All"] + source_options_real
selected_source = st.sidebar.selectbox(
    "Fuente",
    options=source_options,
    index=0
)

language_options_real = sorted([x for x in df["original_language"].dropna().unique().tolist() if x != ""])
language_options = ["All"] + language_options_real
selected_languages = st.sidebar.multiselect(
    "Idiomas",
    options=language_options,
    default=["All"]
)

available_topics = [TOPIC_LABELS[c] for c in TOPIC_COLS if c in df.columns]
selected_topics = st.sidebar.multiselect(
    "Temas",
    options=available_topics,
    default=[]
)

if "release_year" in df.columns and df["release_year"].notna().any():
    min_year = int(df["release_year"].dropna().min())
    max_year = int(df["release_year"].dropna().max())
    year_range = st.sidebar.slider(
        "Rango de años",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_range = None

st.sidebar.divider()
st.sidebar.markdown("### 🔎 Title Quick Search")

sidebar_title_options = sorted(df["title"].dropna().astype(str).unique().tolist())
sidebar_selected_title = st.sidebar.selectbox(
    "Busca un título",
    options=sidebar_title_options,
    index=None,
    placeholder="Empieza a escribir un título..."
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

selected_topic_cols = [
    col for col, label in TOPIC_LABELS.items() if label in selected_topics
]
for col in selected_topic_cols:
    if col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col] == 1]

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR QUICK TITLE RESULT
# ──────────────────────────────────────────────────────────────────────────────
if sidebar_selected_title:
    sidebar_row = df[df["title"] == sidebar_selected_title].head(1)

    if not sidebar_row.empty:
        row = sidebar_row.iloc[0]

        st.sidebar.markdown("#### Información del título")
        st.sidebar.markdown(
            f"**Tipo:** {row.get('content_type', 'N/A')}  \n"
            f"**Año:** {int(row['release_year']) if pd.notna(row.get('release_year')) else 'N/A'}  \n"
            f"**Idioma:** {row.get('original_language', 'N/A')}"
        )

        st.sidebar.markdown("#### Overview")
        overview_text = row.get("overview", "")
        if overview_text:
            st.sidebar.caption(overview_text[:350] + ("..." if len(overview_text) > 350 else ""))
        else:
            st.sidebar.caption("No disponible.")

        st.sidebar.markdown("#### Marketing performance")
        st.sidebar.metric("Visibility", metric_str(row.get("visibility_score", np.nan)))
        st.sidebar.metric("Engagement", metric_str(row.get("engagement_score", np.nan)))
        st.sidebar.metric("Business Value", metric_str(row.get("business_value_score", np.nan)))
        if "predicted_business_value" in row.index:
            st.sidebar.metric("Predicted BV", metric_str(row.get("predicted_business_value", np.nan)))

st.markdown(f"### Dataset filtrado: {filtered_df.shape[0]:,} títulos")

# ──────────────────────────────────────────────────────────────────────────────
# KPIS
# ──────────────────────────────────────────────────────────────────────────────
total_titles = len(filtered_df)
avg_popularity = filtered_df["popularity"].mean() if "popularity" in filtered_df.columns else np.nan
avg_vote = filtered_df["vote_average"].mean() if "vote_average" in filtered_df.columns else np.nan
avg_business = filtered_df["business_value_score"].mean() if "business_value_score" in filtered_df.columns else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total títulos", f"{total_titles:,}")
k2.metric("Popularidad promedio", f"{avg_popularity:.2f}" if pd.notna(avg_popularity) else "N/A")
k3.metric("Rating promedio", f"{avg_vote:.2f}" if pd.notna(avg_vote) else "N/A")
k4.metric("Business Value promedio", f"{avg_business:.2f}" if pd.notna(avg_business) else "N/A")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Marketing Performance",
    "🔎 Similar Titles Finder",
    "🧠 Synopsis Matcher",
    "🗂 Dataset Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Vista general del catálogo")

    c1, c2 = st.columns(2)

    with c1:
        if "content_type" in filtered_df.columns and not filtered_df.empty:
            counts = filtered_df["content_type"].value_counts()
            fig, ax = make_dark_fig((7, 4))
            ax.bar(counts.index, counts.values, color=ACCENT_1)
            ax.set_title("Títulos por tipo de contenido", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("Cantidad", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "source" in filtered_df.columns and not filtered_df.empty:
            counts = filtered_df["source"].value_counts()
            fig, ax = make_dark_fig((7, 4))
            ax.bar(counts.index, counts.values, color=ACCENT_2)
            ax.set_title("Títulos por fuente", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("Cantidad", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    c3, c4 = st.columns(2)

    with c3:
        if "popularity" in filtered_df.columns and filtered_df["popularity"].notna().any():
            fig, ax = make_dark_fig((7, 4))
            ax.hist(filtered_df["popularity"].dropna(), bins=30, color=ACCENT_3)
            ax.set_title("Distribución de popularidad", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Popularity", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c4:
        if "vote_average" in filtered_df.columns and filtered_df["vote_average"].notna().any():
            fig, ax = make_dark_fig((7, 4))
            ax.hist(filtered_df["vote_average"].dropna(), bins=30, color=ACCENT_1)
            ax.set_title("Distribución de rating", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Vote Average", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MARKETING PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Análisis de marketing y performance")

    metric_map = {
        "Visibility": "visibility_score",
        "Engagement": "engagement_score",
        "Business Value": "business_value_score"
    }

    selected_metric_label = st.radio(
        "Métrica destacada:",
        options=list(metric_map.keys()),
        horizontal=True,
        index=2
    )
    highlight_metric = metric_map[selected_metric_label]

    c1, c2 = st.columns(2)

    with c1:
        if highlight_metric in filtered_df.columns and "title" in filtered_df.columns and not filtered_df.empty:
            top_df = filtered_df[["title", highlight_metric]].dropna().sort_values(
                by=highlight_metric, ascending=False
            ).head(10)

            fig, ax = make_dark_fig((8, 5))
            ax.barh(top_df["title"][::-1], top_df[highlight_metric][::-1], color=ACCENT_1)
            ax.set_title(f"Top 10 por {selected_metric_label}", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Score", color=TEXT)
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "marketing_segment" in filtered_df.columns and not filtered_df.empty:
            seg = filtered_df["marketing_segment"].value_counts()
            fig, ax = make_dark_fig((8, 5))
            ax.bar(seg.index, seg.values, color=ACCENT_2)
            ax.set_title("Distribución por marketing segment", color=TEXT, fontsize=13, fontweight="bold", pad=10)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)
            plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SIMILAR TITLES FINDER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Buscador de títulos similares")

    available_titles = sorted(filtered_df["title"].dropna().astype(str).unique().tolist())

    if not available_titles:
        st.warning("No hay títulos disponibles con los filtros actuales.")
    else:
        selected_title = st.selectbox(
            "Empieza a escribir y selecciona un título",
            options=available_titles,
            index=None,
            placeholder="Escribe el nombre de una película o serie..."
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
                info1.metric("Tipo", row.get("content_type", "N/A"))
                info2.metric("Año", int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A")
                info3.metric("Rating", f"{row['vote_average']:.2f}" if pd.notna(row.get("vote_average")) else "N/A")
                info4.metric("Business Value", f"{row['business_value_score']:.2f}" if pd.notna(row.get("business_value_score")) else "N/A")

                st.markdown(
                    f"**Géneros:** {row.get('genre_names', 'N/A')}  \n"
                    f"**Idioma:** {row.get('original_language', 'N/A')}  \n"
                    f"**Marketing Segment:** {row.get('marketing_segment', 'N/A')}  \n"
                    f"**Cluster:** {row.get('cluster_label', 'N/A')}"
                )

                active_topics = get_active_topics_from_row(row)
                st.markdown(f"**Temas detectados:** {', '.join(active_topics) if active_topics else 'No detectados'}")

                if row.get("overview"):
                    st.markdown("**Overview**")
                    st.write(row["overview"])

                r1, r2, r3 = st.columns(3)
                r1.metric("Ranking popularidad", f"#{rank_pop}" if rank_pop else "N/A", delta=f"de {total_pop}" if total_pop else None)
                r2.metric("Ranking rating", f"#{rank_vote}" if rank_vote else "N/A", delta=f"de {total_vote}" if total_vote else None)
                r3.metric("Ranking business value", f"#{rank_business}" if rank_business else "N/A", delta=f"de {total_business}" if total_business else None)

                st.divider()

                similar_df = get_similar_titles(df_similarity, sim_matrix, selected_title, top_n=10)

                st.markdown("#### Títulos similares")

                if similar_df.empty:
                    st.info("No se encontraron títulos similares.")
                else:
                    fig, ax = make_dark_fig((9, 5))
                    plot_df = similar_df.head(8).copy()
                    ax.barh(plot_df["title"][::-1], plot_df["similarity_score"][::-1], color=ACCENT_1)
                    ax.set_title("Top similitud", color=TEXT, fontsize=13, fontweight="bold", pad=10)
                    ax.set_xlabel("Similarity Score", color=TEXT)
                    st.pyplot(fig)
                    plt.close(fig)

                    display_df = similar_df.copy()
                    if "similarity_score" in display_df.columns:
                        display_df["similarity_score"] = display_df["similarity_score"].round(3)

                    st.dataframe(display_df, use_container_width=True)

        if not similar_df.empty:
            download_similar_df = similar_df.copy()
            if "similarity_score" in download_similar_df.columns:
                download_similar_df["similarity_score"] = download_similar_df["similarity_score"].round(5)

            csv_similar = download_similar_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Descargar títulos similares",
                data=csv_similar,
                file_name="similar_titles_results.csv",
                mime="text/csv"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – SYNOPSIS MATCHER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Synopsis Matcher")
    st.markdown(
        "Escribe una sinopsis, elige si es película, serie o ambos, "
        "y te mostramos los temas detectados y los títulos más parecidos del dataset."
    )

    user_type = st.selectbox(
        "Tipo de contenido",
        options=["All", "movie", "tv"],
        index=0
    )

    user_synopsis = st.text_area(
        "Escribe la sinopsis",
        height=180,
        placeholder="Ejemplo: Una joven periodista descubre una red de corrupción política mientras intenta salvar su carrera y proteger a su familia..."
    )

    top_n = st.slider("Número de comparables", min_value=3, max_value=10, value=3)

    synopsis_results = pd.DataFrame()
    detected_topics = []

    if st.button("Analizar sinopsis"):
        if not user_synopsis or not user_synopsis.strip():
            st.warning("Por favor, escribe una sinopsis.")
        else:
            detected_topics = detect_topics_from_synopsis(user_synopsis)

            st.markdown("### Temas detectados")
            st.write(", ".join(detected_topics))

            synopsis_results = get_top_similar_titles(
                user_synopsis=user_synopsis,
                content_type=user_type,
                df=filtered_df,
                top_n=top_n
            )

            st.session_state["synopsis_results"] = synopsis_results
            st.session_state["user_synopsis_text"] = user_synopsis
            st.session_state["user_synopsis_topics"] = detected_topics
            st.session_state["user_synopsis_type"] = user_type

    if "synopsis_results" in st.session_state:
        synopsis_results = st.session_state["synopsis_results"]
        user_synopsis = st.session_state.get("user_synopsis_text", "")
        detected_topics = st.session_state.get("user_synopsis_topics", [])
        user_type_saved = st.session_state.get("user_synopsis_type", "All")

        if detected_topics:
            st.markdown("### Temas detectados")
            st.write(", ".join(detected_topics))

        st.markdown(f"### Top {len(synopsis_results)} títulos más parecidos")

        if synopsis_results.empty:
            st.info("No se encontraron títulos comparables con esos filtros.")
        else:
            for i, (_, row) in enumerate(synopsis_results.iterrows(), start=1):
                st.markdown(f"#### #{i} - {row.get('title', 'Unknown title')}")
                st.write(f"**Tipo:** {row.get('content_type', 'N/A')}")
                st.write(f"**Géneros:** {row.get('genre_names', 'N/A')}")
                st.write(f"**Idioma:** {row.get('original_language', 'N/A')}")
                st.write(f"**Año:** {row.get('release_year', 'N/A')}")
                st.write(f"**Text similarity:** {row.get('text_similarity', 0):.3f}")
                st.write(f"**Topic similarity:** {row.get('topic_similarity', 0):.3f}")
                st.write(f"**Similarity score:** {row.get('similarity_score', 0):.3f}")

                if "business_value_score" in row and pd.notna(row["business_value_score"]):
                    st.write(f"**Business Value Score:** {row['business_value_score']:.2f}")

                if "predicted_business_value" in row and pd.notna(row["predicted_business_value"]):
                    st.write(f"**Predicted Business Value:** {row['predicted_business_value']:.2f}")

                st.write(f"**Overview:** {row.get('overview', 'N/A')}")
                st.markdown("---")

            avg_bv = synopsis_results["business_value_score"].mean() if "business_value_score" in synopsis_results.columns else np.nan

            st.markdown("### Lectura rápida")
            if pd.notna(avg_bv):
                if avg_bv >= 70:
                    st.success("La idea se parece a contenidos con alto potencial comercial.")
                elif avg_bv >= 45:
                    st.info("La idea se parece a contenidos con potencial medio; podría depender del marketing y posicionamiento.")
                else:
                    st.warning("La idea se parece a contenidos con menor valor comercial estimado en el dataset.")

            export_df = synopsis_results.copy()
            export_df.insert(0, "input_synopsis", user_synopsis)
            export_df.insert(1, "input_content_type", user_type_saved)
            export_df.insert(2, "detected_topics", ", ".join(detected_topics))

            csv_synopsis = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Descargar sinopsis + títulos similares",
                data=csv_synopsis,
                file_name="synopsis_matcher_results.csv",
                mime="text/csv"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Explorador del dataset")

    explorer_cols = [c for c in [
        "title", "content_type", "genre_names", "release_year",
        "original_language", "source", "popularity", "vote_average",
        "vote_count", "visibility_score", "engagement_score",
        "business_value_score", "predicted_business_value",
        "marketing_segment", "cluster_label", "overview"
    ] if c in filtered_df.columns]

    st.dataframe(filtered_df[explorer_cols].head(100), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Descargar dataset filtrado",
        data=csv,
        file_name="filtered_streaming_dataset.csv",
        mime="text/csv"
    )
