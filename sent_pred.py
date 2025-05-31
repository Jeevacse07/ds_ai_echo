import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from datetime import datetime

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ==== Page Config ====
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# ==== Load Assets ====
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\Jeeva\ds_course\sentiment_analysis\chatgpt_reviews.csv", parse_dates=['date'])


@st.cache_data
def load_assets():
    with open(r"C:\Users\JEEVA\sentiment_analysis\categorical_label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(r"C:\Users\JEEVA\sentiment_analysis\tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(r"C:\Users\JEEVA\sentiment_analysis\model_nb.pkl", "rb") as f:
        model_nb = pickle.load(f)
    with open(r"C:\Users\JEEVA\sentiment_analysis\model_lr.pkl", "rb") as f:
        model_lr = pickle.load(f)
    with open(r"C:\Users\JEEVA\sentiment_analysis\model_rf.pkl", "rb") as f:
        model_rf = pickle.load(f)
    return label_encoders, tfidf, model_nb, model_lr, model_rf

df = load_data()
label_encoders, tfidf, model_nb, model_lr, model_rf = load_assets()

# ==== Data Preparation ====
def map_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["rating"].apply(map_sentiment)

# ==== TAB LAYOUT ====
tab1, tab2 = st.tabs(["📊 EDA Dashboard", "🔍 Predict Sentiment"])

# ==== TAB 1: EDA ====
with tab1:
   # Sidebar navigation
    st.sidebar.title("EDA Dashboard")
    section = st.sidebar.radio("Select Analysis", [
        "Rating Distribution", "Helpful Votes", "Word Clouds",
        "Time Series Trend", "Location-Based Sentiment",
        "Platform Comparison", "Verified Purchase Analysis",
        "Review Length vs Sentiment", "Version Analysis"
    ])

    # Section 1: Rating Distribution
    if section == "Rating Distribution":
        st.header("Rating Distribution Overview")
    
        # Histogram + Bar Chart
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Histogram: Rating Distribution")
            fig_hist = px.histogram(df, x="rating", nbins=5, color="rating",
                                    color_discrete_sequence=px.colors.sequential.Blues,
                                    title="Rating Histogram")
            st.plotly_chart(fig_hist, use_container_width=True)
    
        with col2:
            st.subheader("Bar Chart: Rating Counts")
            rating_counts = df["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            fig_bar = px.bar(rating_counts, x="rating", y="count", text_auto=True,
                             color="rating", title="Number of Ratings per Value",
                             color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_bar, use_container_width=True)
    
        # Pie Charts: Rating and Sentiment
        col3, col4 = st.columns(2)
    
        with col3:
            st.subheader("Donut Chart: Votes by Rating")
            fig_pie = px.pie(rating_counts, names="rating", values="count", hole=0.4,
                             title="Proportion of Ratings",
                             color_discrete_map={'1':'lightcyan','2':'cyan','3':'royalblue','4':'darkblue','5':'cornflowerblue'})
            st.plotly_chart(fig_pie, use_container_width=True)
    
        with col4:
            st.subheader("Donut Chart: Votes by Sentiment")
            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig_pie_sentiment = px.pie(sentiment_counts, names="sentiment", values="count", hole=0.4,
                                       title="Proportion of Sentiments",
                                       color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie_sentiment, use_container_width=True)
    
        # Visual Comparison: 1-Star vs 5-Star
        st.markdown("### ⭐ Visual Comparison: 1-Star vs 5-Star Ratings")
        count_1_star = df[df["rating"] == 1].shape[0]
        count_5_star = df[df["rating"] == 5].shape[0]
    
        col5, col6 = st.columns(2)
        with col5:
            fig_1 = px.pie(values=[count_1_star, rating_counts["count"].sum() - count_1_star],
                           names=["1 Star", "Others"], hole=0.5,
                           title="1-Star Share",
                           color_discrete_sequence=["crimson", "lightgrey"])
            st.plotly_chart(fig_1, use_container_width=True)
        with col6:
            fig_5 = px.pie(values=[count_5_star, rating_counts["count"].sum() - count_5_star],
                           names=["5 Star", "Others"], hole=0.5,
                           title="5-Star Share",
                           color_discrete_sequence=["green", "lightgrey"])
            st.plotly_chart(fig_5, use_container_width=True)
    
        # Sunburst for Rating by Location
        st.subheader("Sunburst: Rating Distribution by Location")
        rating_location = df.groupby(["location", "rating"]).size().reset_index(name="count")
        fig_sunburst_loc = px.sunburst(rating_location, path=["location", "rating"], values="count",
                                       title="Ratings Nested within Locations",
                                       color="rating", color_continuous_scale="blues")
        st.plotly_chart(fig_sunburst_loc, use_container_width=True)
    
        # Grouped Bar Chart for Rating by Language
        st.subheader("Grouped Bar: Rating Count by Language")
        rating_lang = df.groupby(["language", "rating"]).size().reset_index(name="count")
        fig_lang_grouped = px.bar(rating_lang, x="language", y="count", color="rating",
                                  title="Ratings Grouped by Language",
                                  barmode="group", text_auto=True,
                                  color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig_lang_grouped, use_container_width=True)
    
    
    
    
    # Section 2: Helpful Votes
    elif section == "Helpful Votes":
        st.header("Helpful Votes Analysis")
        st.metric("Total Helpful Votes", int(df.helpful_votes.sum()))
        top_helpful = df.sort_values(by=["helpful_votes"], ascending=False).head(10)
    
        st.write("Top 10 Helpful Reviews")
        st.dataframe(top_helpful[["title", "review", "rating", "helpful_votes"]])
    
        st.header("Helpful Votes by Sentiment")
    
        # Aggregate helpful votes by sentiment
        sentiment_helpful = df.groupby("sentiment")["helpful_votes"].sum().sort_values(ascending=False).reset_index()
    
        st.write("Total Helpful Votes per Sentiment")
        st.dataframe(sentiment_helpful)
    
        fig_sentiment = px.bar(
            sentiment_helpful,
            x="sentiment",
            y="helpful_votes",
            color="sentiment",
            text_auto=True,
            title="Helpful Votes by Sentiment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
        # Chart 1: Helpful Votes by Country
        st.header("Helpful Votes by Country")
        country_helpful = df.groupby("location")["helpful_votes"].sum().sort_values(ascending=False).reset_index()
        st.dataframe(country_helpful)
    
        fig_country = px.bar(
            country_helpful,
            x="location",
            y="helpful_votes",
            color="location",
            text_auto=True,
            title="Helpful Votes by Country",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_country, use_container_width=True)
    
        # Chart 2: Helpful Votes by Language
        st.header("Helpful Votes by Language")
        language_helpful = df.groupby("language")["helpful_votes"].sum().sort_values(ascending=False).reset_index()
        st.dataframe(language_helpful)
    
        fig_language = px.bar(
            language_helpful,
            x="language",
            y="helpful_votes",
            color="language",
            text_auto=True,
            title="Helpful Votes by Language",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_language, use_container_width=True)
    
        # Chart: Sentiment Count by Country
        st.header("Sentiment Distribution by Location")
        sentiment_country = df.groupby(["location", "sentiment"]).size().unstack(fill_value=0)
        st.dataframe(sentiment_country)
    
        sentiment_country_reset = sentiment_country.reset_index().melt(id_vars="location", var_name="sentiment", value_name="count")
        fig_sentiment_country = px.bar(
            sentiment_country_reset,
            x="location",
            y="count",
            color="sentiment",
            barmode="group",
            text_auto=True,
            title="Sentiment Distribution by Country",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        st.plotly_chart(fig_sentiment_country, use_container_width=True)
    
        # Chart: Sentiment Distribution by Language
        st.header("Sentiment Distribution by Language")
        sentiment_language = df.groupby(["language", "sentiment"]).size().unstack(fill_value=0)
        st.dataframe(sentiment_language)
    
        sentiment_language_reset = sentiment_language.reset_index().melt(id_vars="language", var_name="sentiment", value_name="count")
        fig_sentiment_language = px.bar(
            sentiment_language_reset,
            x="language",
            y="count",
            color="sentiment",
            barmode="group",
            text_auto=True,
            title="Sentiment Distribution by Language",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig_sentiment_language, use_container_width=True)
    
    
    # Section 3: Word Clouds
    elif section == "Word Clouds":
        st.header("Word Clouds and Top Words for Each Sentiment")
    
        stop_words = set(stopwords.words('english'))
    
        for sentiment in ["positive", "negative"]:
            st.subheader(f"{sentiment.title()} Reviews")
    
            # Combine text
            text = " ".join(df[df.sentiment == sentiment].review.astype(str)).lower()
    
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
    
            # Preprocess words
            words = [word for word in text.split() if word.isalpha() and word not in stop_words]
            word_counts = Counter(words)
            top_words = word_counts.most_common(10)
            words_df = pd.DataFrame(top_words, columns=["word", "count"])
    
            # Plot top 10 words
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            sns.barplot(data=words_df, y="word", x="count", ax=ax_bar, palette="viridis")
            ax_bar.set_xlabel("Count")
            ax_bar.set_ylabel("Word")
            ax_bar.set_title(f"Top 10 Words - {sentiment.title()} Sentiment")
            st.pyplot(fig_bar)
    
    
    # Section 4: Time Series Trend
    elif section == "Time Series Trend":
        st.header("Time-Series Trend of Average Rating")
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        avg_rating = df.groupby("month_year")["rating"].mean().reset_index()
        fig = px.line(avg_rating, x="month_year", y="rating", markers=True,
                      title="Average Rating Over Time")
        st.plotly_chart(fig)
    
    
        pos_daily = df[df['sentiment'] == 'positive'].groupby('month_year').size().reset_index(name='count')
        fig_pos = px.line(pos_daily, x="month_year", y="count", markers=True,title="Positive Review Trend Over Time")
        st.plotly_chart(fig_pos)
        
        neg_daily = df[df['sentiment'] == 'negative'].groupby('month_year').size().reset_index(name='count')
        fig_neg = px.line(neg_daily, x="month_year", y="count", markers=True,title="Negative Review Trend Over Time")
        st.plotly_chart(fig_neg)
    
        monthly_counts = df.groupby(['month_year', 'sentiment']).size().unstack(fill_value=0).reset_index()
        monthly_counts['pos_neg_ratio'] = monthly_counts['positive'] / (monthly_counts['negative'] + 1)
    
        fig_ratio = px.line(monthly_counts, x='month_year', y='pos_neg_ratio',
                        title='Positive-to-Negative Sentiment Ratio Over Time', markers=True)
        st.plotly_chart(fig_ratio)
    
        monthly_counts['pos_pct_change'] = monthly_counts['positive'].pct_change() * 100
        monthly_counts['neg_pct_change'] = monthly_counts['negative'].pct_change() * 100
    
        fig_vol = px.line(monthly_counts, x='month_year', y=['pos_pct_change', 'neg_pct_change'],
                      title='Month-over-Month Sentiment Volatility (%)', markers=True)
        st.plotly_chart(fig_vol)
    
        monthly_counts['pos_ma'] = monthly_counts['positive'].rolling(3).mean()
        monthly_counts['neg_ma'] = monthly_counts['negative'].rolling(3).mean()
    
        fig_ma = px.line(monthly_counts, x='month_year', y=['pos_ma', 'neg_ma'],
                     title='3-Month Moving Average of Sentiment', markers=True)
        st.plotly_chart(fig_ma)
    
    
    
    # Section 5: Location-Based Sentiment
    
    elif section == "Location-Based Sentiment":
        st.header("Location-Based Sentiment")
    
        # 1. Grouped (side-by-side) bar chart: Positive vs Negative per location
        loc_sent = df.groupby(["location", "sentiment"]).size().reset_index(name='count')
        loc_sent_filtered = loc_sent[loc_sent['sentiment'].isin(['positive', 'negative'])]
        
        st.subheader("Side-by-Side Bar Chart: Positive vs Negative Sentiment per Location")
        fig = px.bar(loc_sent_filtered, 
                     x="location", 
                     y="count", 
                     color="sentiment", 
                     barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     title="Positive vs Negative Sentiment Count by Location")
        st.plotly_chart(fig)
    
        # 2. Total posts per location
        st.subheader("Total Posts per Location")
        total_posts = df['location'].value_counts().reset_index()
        total_posts.columns = ['Location', 'Total Posts']
        st.dataframe(total_posts)
    
        # 3. Top 3 locations with most positive sentiment posts (Chart)
        st.subheader("Top 3 Locations with Most Positive Sentiment Posts")
        pos_counts = df[df["sentiment"] == "positive"]["location"].value_counts().head(3).reset_index()
        pos_counts.columns = ['Location', 'Positive Posts']
        fig_pos = px.bar(pos_counts, x="Location", y="Positive Posts", color="Location", text="Positive Posts",
                         title="Top 3 Locations - Positive Sentiment",color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pos)
    
        # 4. Top 3 locations with most negative sentiment posts (Chart)
        st.subheader("Top 3 Locations with Most Negative Sentiment Posts")
        neg_counts = df[df["sentiment"] == "negative"]["location"].value_counts().head(3).reset_index()
        neg_counts.columns = ['Location', 'Negative Posts']
        fig_neg = px.bar(neg_counts, x="Location", y="Negative Posts", color="Location", text="Negative Posts",
                         title="Top 3 Locations - Negative Sentiment",color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_neg)
    
        # 5. Sentiment percentage distribution per location
        st.subheader("Sentiment Percentage Distribution per Location")
        loc_sent_pivot = df.groupby(["location", "sentiment"]).size().unstack(fill_value=0)
        loc_sent_pct = loc_sent_pivot.div(loc_sent_pivot.sum(axis=1), axis=0).round(2) * 100
        st.dataframe(loc_sent_pct)
    
        # 6. Pie chart: Total posts per location
        with st.expander("📊 Pie Chart of Total Posts per Location"):
            st.write("Visual breakdown of total posts per location")
            st.plotly_chart(px.pie(total_posts, names='Location', values='Total Posts', title='Total Posts per Location'))
    
        
    
    # Section 6: Platform Comparison
    elif section == "Platform Comparison":
        st.header("Platform vs Sentiment")
    
        # Platform vs Sentiment
        plat_df = df.groupby(["platform", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(plat_df, x="platform", y="count", color="sentiment",
                     title="Platform-wise Sentiment Distribution",
                     barmode="group", color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig)
    
        # Platform Usage by Location
        loc_plat_df = df.groupby(["location", "platform"]).size().reset_index(name="count")
        fig = px.bar(loc_plat_df, x="count", y="location", color="platform", orientation="h",
                     title="Platform Usage by Location", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
    
        # Language Usage by Platform — FIXED
        lang_plat_df = df.groupby(["language", "platform"]).size().reset_index(name="count")
        fig = px.bar(lang_plat_df, x="count", y="language", color="platform", orientation="h",
                     title="Language Usage by Platform", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(texttemplate='%{x}', textposition='outside')
        st.plotly_chart(fig)
    
    
        # Positive Sentiment Count
        positive_df = df[df["sentiment"] == "positive"]
        pos_counts = positive_df["platform"].value_counts().reset_index()
        pos_counts.columns = ["platform", "positive_count"]
        fig_pos = px.bar(pos_counts, x="platform", y="positive_count", 
                         title="Positive Sentiment Count by Platform",
                         color_discrete_sequence=["forestgreen"])
        fig_pos.update_traces(texttemplate='%{y}', textposition='outside')
    
        # Negative Sentiment Count
        negative_df = df[df["sentiment"] == "negative"]
        neg_counts = negative_df["platform"].value_counts().reset_index()
        neg_counts.columns = ["platform", "negative_count"]
        fig_neg = px.bar(neg_counts, x="platform", y="negative_count", 
                         title="Negative Sentiment Count by Platform",
                         color_discrete_sequence=["lightsalmon"])
        fig_neg.update_traces(texttemplate='%{y}', textposition='outside')
    
        # Show side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pos, use_container_width=True)
        with col2:
            st.plotly_chart(fig_neg, use_container_width=True)
    
        # Sentiment % Distribution per Platform
        sentiment_ratio = plat_df.copy()
        total_by_platform = sentiment_ratio.groupby("platform")["count"].transform("sum")
        sentiment_ratio["percentage"] = (sentiment_ratio["count"] / total_by_platform * 100).round(2)
        fig = px.bar(sentiment_ratio, x="platform", y="percentage", color="sentiment",
                     title="Sentiment % Distribution by Platform", barmode="stack",
                     color_discrete_sequence=["gold","silver","yellowgreen"])
        fig.update_traces(texttemplate='%{y}%', textposition='inside')
        st.plotly_chart(fig)
    
        # Top 3 Locations by Positive Sentiment
        loc_sent_df = df[df["sentiment"] == "positive"].groupby("location").size().reset_index(name="positive_count")
        top_pos_locs = loc_sent_df.sort_values("positive_count", ascending=False).head(3)
        fig = px.bar(top_pos_locs, x="positive_count", y="location", orientation="h",
                     title="Top 3 Locations by Positive Sentiment", color_discrete_sequence=["hotpink"])
        st.plotly_chart(fig)
    
        # Top 3 Locations by Negative Sentiment
        loc_sent_df = df[df["sentiment"] == "negative"].groupby("location").size().reset_index(name="negative_count")
        top_neg_locs = loc_sent_df.sort_values("negative_count", ascending=False).head(3)
        fig = px.bar(top_neg_locs, x="negative_count", y="location", orientation="h",
                     title="Top 3 Locations by Negative Sentiment", color_discrete_sequence=["orange"])
        st.plotly_chart(fig)
    
        # Platform Location Diversity
        diversity_df = df.groupby("platform")["location"].nunique().reset_index(name="unique_locations")
        fig = px.bar(diversity_df, x="platform", y="unique_locations",
                     title="Platform Location Diversity Score",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig)
    
    
        pie_df_pos = plat_df[plat_df["sentiment"] == "positive"]
        fig_pos = px.pie(pie_df_pos, names="platform", values="count",
                     title="Positive Sentiment Share by Platform",color_discrete_sequence=["green", "blue"])
        st.plotly_chart(fig_pos)
    
        pie_df_neg = plat_df[plat_df["sentiment"] == "negative"]
        fig_neg = px.pie(pie_df_neg, names="platform", values="count",
                     title="Negative Sentiment Share by Platform",color_discrete_sequence=["tomato", "violet"])
        st.plotly_chart(fig_neg)
    
    
    # Section 7: Verified Purchase Analysis
    elif section == "Verified Purchase Analysis":
        st.header("Verified vs Non-Verified Sentiment")
        ver_df = df.groupby(["verified_purchase", "sentiment"]).size().reset_index(name="count")
        fig1 = px.bar(ver_df, x="verified_purchase", y="count", color="sentiment",
                     title="Sentiment by Verified Purchase", color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig1)
    
        # Verified vs Unverified Purchase by Location
        st.subheader("Top 3 Locations: Verified vs Unverified Purchases")
        ver_loc = df[df["verified_purchase"] == "Yes"].groupby("location").size().reset_index(name="verified_count")
        unver_loc = df[df["verified_purchase"] == "No"].groupby("location").size().reset_index(name="unverified_count")
        
        ver_loc = ver_loc.sort_values("verified_count", ascending=False).head(3)
        unver_loc = unver_loc.sort_values("unverified_count", ascending=False).head(3)
    
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.bar(ver_loc, x="verified_count", y="location", orientation='h',
                          title="Top 3 Locations - Verified", color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig2)
        with col2:
            fig3 = px.bar(unver_loc, x="unverified_count", y="location", orientation='h',
                          title="Top 3 Locations - Unverified", color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig3)
    
        # Verified Purchase by Language
        st.subheader("Verified Purchase Count by Language")
        ver_lang = df[df["verified_purchase"] == "Yes"].groupby("language").size().reset_index(name="verified_count")
        ver_lang = ver_lang.sort_values("verified_count", ascending=False)
        fig4 = px.bar(ver_lang, x="language", y="verified_count", title="Verified Purchases by Language",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig4)
    
        # Verified Sentiment Breakdown
        st.subheader("Sentiment Breakdown - Verified Purchases Only")
        ver_sent = df[df["verified_purchase"] == "Yes"].groupby("sentiment").size().reset_index(name="count")
        fig5 = px.pie(ver_sent, names="sentiment", values="count", title="Sentiment in Verified Purchases",
                      color_discrete_sequence=px.colors.sequential.Agsunset)
        st.plotly_chart(fig5)
    
        # Verified Purchases Over Time
        st.subheader("Verified Purchase Trend Over Time")
        df['review_date'] = pd.to_datetime(df['date'])
        ver_time = df[df["verified_purchase"] == "Yes"].groupby(df['review_date'].dt.to_period('M')).size().reset_index(name="verified_count")
        ver_time['review_date'] = ver_time['review_date'].astype(str)
        fig6 = px.line(ver_time, x="review_date", y="verified_count", markers=True,
                       title="Verified Purchases Over Time", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig6)
    
        # Verified Purchase by Platform
        st.subheader("Verified Purchases by Platform")
        ver_platform = df[df["verified_purchase"] == "Yes"].groupby("platform").size().reset_index(name="verified_count")
        ver_platform = ver_platform.sort_values("verified_count", ascending=False)
        fig7 = px.bar(ver_platform, x="platform", y="verified_count", title="Verified Purchases by Platform",
                      color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig7)
    
        # Verified Purchase Ratio by Location
        st.subheader("Verified Purchase Ratio by Location")
        location_ver_ratio = df.groupby("location")["verified_purchase"].apply(lambda x: (x=="Yes").sum() / len(x)).reset_index(name="verified_ratio")
        location_ver_ratio = location_ver_ratio.sort_values("verified_ratio", ascending=False).head(3)
        fig8 = px.bar(location_ver_ratio, x="location", y="verified_ratio",
                      title="Top 3 Locations by Verified Purchase Ratio", color_discrete_sequence=px.colors.sequential.Magma)
        st.plotly_chart(fig8)
    
        # Daily Verified Reviews
        st.subheader("Daily Verified Reviews")
        ver_daily = df[df["verified_purchase"] == "Yes"].groupby(df["review_date"].dt.date).size().reset_index(name="daily_verified_count")
        fig9 = px.area(ver_daily, x="review_date", y="daily_verified_count",
                       title="Verified Reviews Per Day", color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig9)
    
    
    # Section 8: Review Length vs Sentiment
    elif section == "Review Length vs Sentiment":
    
        # Review Length vs Sentiment (Box Plot)
        st.header("Review Length vs Sentiment")
        fig1 = px.box(df, x="sentiment", y="review_length", color="sentiment",
                      title="Review Length by Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1)
        
        # Review Length vs Location (Violin Plot)
        st.header("Review Length vs Location")
        fig2 = px.violin(df, x="location", y="review_length", color="location", box=True, points="all",
                         title="Review Length by Location", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2)
        
        # Review Length vs Language (Strip Plot)
        st.header("Review Length vs Language")
        fig3 = px.strip(df, x="language", y="review_length", color="language",
                        title="Review Length by Language", color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig3)
        
        # Review Length vs Verified Purchase (Histogram)
        st.header("Review Length vs Verified Purchase")
        fig4 = px.histogram(df, x="review_length", color="verified_purchase", barmode="overlay",
                            title="Review Length Distribution: Verified vs Non-Verified", nbins=50,
                            color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig4)
        
        # Review Length vs Version (Swarm Dot Plot)
        st.header("Review Length vs App Version")
        fig5 = px.strip(df, x="version", y="review_length", color="version",
                        title="Review Length by App Version", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig5)
        
        # Review Length Over Time (Line Plot)
        st.header("Review Length Over Time")
        df_time = df.copy()
        df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
        avg_length_by_month = df_time.groupby("month")["review_length"].mean().reset_index()
        fig6 = px.line(avg_length_by_month, x="month", y="review_length",
                       title="Average Review Length Over Time", markers=True,
                       line_shape="spline", color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig6)
    
    
    # Section 9: Version vs Rating
    elif section == "Version Analysis":
        st.header("Version vs Other Attributes Analysis")
        df["date"] = pd.to_datetime(df["date"])  # Ensure proper datetime format
    
        st.subheader("1. Average Rating per Version")
        fig1 = px.bar(df.groupby("version")["rating"].mean().reset_index(),
                      x="version", y="rating", text_auto=True,
                      color="version", title="Average Rating per Version",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig1, use_container_width=True)
    
        st.subheader("2. Sentiment Distribution per Version")
        fig2 = px.histogram(df, x="version", color="sentiment", barmode="group",
                            title="Sentiment Count per Version",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
    
        st.subheader("3. Language Distribution per Version")
        fig3 = px.histogram(df, x="version", color="language", barmode="group",
                            title="Languages per Version",
                            color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig3, use_container_width=True)
    
        st.subheader("4. Platform Usage per Version")
        fig4 = px.histogram(df, x="version", color="platform", barmode="group",
                            title="Platform Distribution by Version",
                            color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig4, use_container_width=True)
    
        st.subheader("5. Verified Purchases per Version")
        fig5 = px.histogram(df, x="version", color="verified_purchase", barmode="group",
                            title="Verified Purchase Distribution by Version",
                            color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig5, use_container_width=True)
    
        st.subheader("6. Location Distribution per Version")
        fig6 = px.histogram(df, x="version", color="location", barmode="stack",
                            title="User Locations per Version",
                            color_discrete_sequence=px.colors.diverging.Temps)
        st.plotly_chart(fig6, use_container_width=True)
    
        st.subheader("7. Ratings Over Time by Version")
        time_df = df.groupby(["date", "version"])["rating"].mean().reset_index()
        fig7 = px.line(time_df, x="date", y="rating", color="version",
                       title="Average Rating Over Time by Version",
                       color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig7, use_container_width=True)
    

# ==== TAB 2: MODEL PREDICTION ====
with tab2:
    st.header("🔍 Predict Sentiment Based on User Input")

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    with st.form("prediction_form"):
        review = st.text_area("Review Text", "This app is fantastic!")
        title = st.text_input("Review Title", "Amazing")
        platform = st.selectbox("Platform", ['Mobile', 'Web'])
        language = st.selectbox("Language", ['en', 'es', 'de', 'hi', 'fr'])
        location = st.selectbox("Location", ['India', 'USA', 'UK', 'Canada', 'Germany', 'Australia'])
        verified = st.selectbox("Verified Purchase", ['Yes', 'No'])
        version = st.selectbox("Version", [3.0, 3.5, 4.0, 4.1])
        rating = st.slider("Rating", 1, 5, 4)
        helpful_votes = st.number_input("Helpful Votes", min_value=0, value=10)

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Predict Sentiment")
        with col2:
            reset = st.form_submit_button("Reset")

    if submit:
        review_len = len(review)
        input_df = pd.DataFrame({
            "platform": [platform],
            "language": [language],
            "location": [location],
            "verified_purchase": [verified],
            "rating": [rating],
            "helpful_votes": [helpful_votes],
            "review_length": [review_len],
            "version": [version] 
        })

        for col in ["platform", "language", "location", "verified_purchase"]:
            input_df[col + "_enc"] = label_encoders[col].transform(input_df[col])

        X_tabular = input_df[[col + "_enc" for col in ["platform", "language", "location", "verified_purchase"]] +
                             ["rating", "helpful_votes", "review_length","version"]]
        X_text = tfidf.transform([review])
        X_final = hstack([X_text, csr_matrix(X_tabular.values)])

        pred_nb = model_nb.predict(X_final)[0]
        pred_lr = model_lr.predict(X_final)[0]
        pred_rf = model_rf.predict(X_final)[0]

        st.success(f"🔵 Naive Bayes: {pred_nb}")
        st.success(f"🟢 Logistic Regression: {pred_lr}")
        st.success(f"🟠 Random Forest: {pred_rf}")

    if reset:
        st.session_state.clear()
        st.rerun()
