🧠 Phase 2: The Analysis Engine

Goal: Turn raw text into actionable data (Sentiment & Topics).
Output: An updated data_loader.py that outputs a DataFrame with "Vibe" and "Topic" columns.
Prerequisites: You must have Phase 1 working (the script prints raw data).

📝 Context

Now that we have the data, we need to process it. We will add two distinct "Brain" functions to your existing data_loader.py file:

VADER (Valence Aware Dictionary and sEntiment Reasoner): A rule-based sentiment analysis tool specifically tuned for social media (perfect for headlines).

BERTopic: A topic modeling technique that uses transformers (like BERT) and clustering to find themes automatically.

🤖 Step-by-Step AI Prompts

Task 2.1: The Sentiment Engine

Action: Open data_loader.py. We will add a new function.
Agent Prompt: (Paste into Cursor/ChatGPT)

"Update data_loader.py to include sentiment analysis.

Imports: Add import nltk and from nltk.sentiment.vader import SentimentIntensityAnalyzer.

Download: Inside the function (or at the top level), ensure nltk.download('vader_lexicon') is run once.

Function: Create analyze_sentiment(df).

Initialize the SentimentIntensityAnalyzer.

Apply it to the title column of the DataFrame.

Extract the compound score into a new column sentiment_score.

Create a sentiment_label column:

'Positive' if score > 0.05

'Negative' if score < -0.05

'Neutral' otherwise.

Return: The modified DataFrame."

Task 2.2: The Topic Engine

Action: Still in data_loader.py, add the topic modeling.
Agent Prompt:

"Add a function get_topics(df) to data_loader.py using the BERTopic library.

Imports: Add from bertopic import BERTopic.

Logic:

Initialize a simple BERTopic model (use embedding_model='all-MiniLM-L6-v2' for speed if possible, or let it default).

Fit the model on the title column converted to a list.

Transform the documents to get topics.

Add the topic ID to a new column topic_id in the DataFrame.

Extract the custom topic labels (keywords) and add a topic_keyword column to the DataFrame so we know what 'Topic 0' actually means.

Return: The modified DataFrame."

Task 2.3: Verification (The "Smart" Sanity Check)

Action: Update the main block to test these new functions.
Agent Prompt:

"Update the if __name__ == '__main__': block at the bottom:

if __name__ == '__main__':
    print('1. Fetching data...')
    df = fetch_hn_data(limit=20) # Fetch 20 for a good sample


print('2. Analyzing Sentiment...')
df = analyze_sentiment(df)


print('3. Extracting Topics...')
df = get_topics(df)


print('\nTop 5 Rows with Analysis:')
print(df[['title', 'sentiment_label', 'topic_keyword']].head())



Ensure the script runs from start to finish."


✅ Success Criteria

Run the script: python data_loader.py

Win Condition:

It does not crash on nltk or bertopic imports.

The final print output shows columns for sentiment (e.g., "Positive") and topic (e.g., "AI_LLM_Model").

Note: The first run might take 30-60 seconds to download the BERT models.

🐛 Troubleshooting

Error: RuntimeError: NLTK data not found

Fix: Make sure nltk.download('vader_lexicon') is actually in your code.

Error: Visual C++ Build Tools or Hdbscan errors

Fix: BERTopic relies on hdbscan. If this fails on Windows, try installing the pre-compiled binary: pip install hdbscan --no-build-isolation or ask the AI: "Help me install BERTopic on Windows without build tools."