import pandas as pd
from textblob import TextBlob
import os
from config import SENTIMENT_PATH, DATA_DIR

# ── Sample Headlines ──────────────────────────────────────────────────────────
# In production replace with a NewsAPI / RSS feed call.
HEADLINES = [
    {"date": "2020-01-01", "headline": "Infosys reports strong quarterly earnings beating estimates"},
    {"date": "2020-01-02", "headline": "Infosys wins major cloud deal with European client"},
    {"date": "2020-01-03", "headline": "Infosys faces pressure as rupee strengthens against dollar"},
    {"date": "2020-01-06", "headline": "Infosys CEO confident about growth outlook for 2020"},
    {"date": "2020-01-07", "headline": "Infosys stock falls amid global market uncertainty"},
    {"date": "2020-03-15", "headline": "Infosys warns of slowdown due to COVID-19 pandemic impact"},
    {"date": "2020-06-01", "headline": "Infosys raises revenue guidance on strong deal wins"},
    {"date": "2021-01-13", "headline": "Infosys Q3 results exceed expectations profit surges"},
    {"date": "2021-07-14", "headline": "Infosys raises FY22 revenue growth guidance to 14-16 percent"},
    {"date": "2022-01-12", "headline": "Infosys reports highest ever quarterly revenue in Q3"},
    {"date": "2022-04-13", "headline": "Infosys misses revenue estimates stock drops sharply"},
    {"date": "2023-01-11", "headline": "Infosys cautious on outlook citing weak demand from clients"},
    {"date": "2023-07-20", "headline": "Infosys cuts FY24 revenue growth forecast amid slowdown"},
    {"date": "2024-01-11", "headline": "Infosys raises guidance after strong deal pipeline growth"},
    {"date": "2024-07-18", "headline": "Infosys beats Q1 estimates raises annual revenue forecast"},
]

def analyze_sentiment(headlines):
    results = []
    for item in headlines:
        blob         = TextBlob(item['headline'])
        polarity     = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label = (
            "Positive" if polarity >  0.05 else
            "Negative" if polarity < -0.05 else
            "Neutral"
        )
        results.append({
            "date":         item['date'],
            "headline":     item['headline'],
            "polarity":     round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "sentiment":    label,
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = analyze_sentiment(HEADLINES)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SENTIMENT_PATH, index=False)

    print("✅ Sentiment analysis complete!")
    print(f"   Positive : {len(df[df['sentiment'] == 'Positive'])}")
    print(f"   Negative : {len(df[df['sentiment'] == 'Negative'])}")
    print(f"   Neutral  : {len(df[df['sentiment'] == 'Neutral'])}")
    print("\n📊 Sample output:")
    print(df[['date', 'sentiment', 'polarity', 'headline']].to_string(index=False))