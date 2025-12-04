🔮 Phase 7: The Oracle (Virality Prediction)

Goal: Build a Machine Learning model to predict if a post will be popular.
Output: A sidebar widget where you type a headline, and the AI gives you a "Viral Probability" score.
Prerequisites: You must have Phase 3 working.

📝 Context

Data Science isn't just about analyzing the past; it's about predicting the future.
We will train a Random Forest Classifier (using scikit-learn) to look at patterns in high-scoring posts.

The Hypothesis:

Do positive titles get more votes?

Do longer titles perform better?

Does posting at a certain hour matter?

We will teach the model to find these patterns.

🤖 Step-by-Step AI Prompts

Task 7.1: The Training Logic

Action: Update data_loader.py to train the model.
Note: You might need to fetch more data (e.g., 100 stories) to get a good model.
Agent Prompt: (Paste into Cursor/ChatGPT)

"I need to add a predictive model to data_loader.py.

Imports: Add from sklearn.ensemble import RandomForestClassifier and from sklearn.model_selection import train_test_split.

Function: Create train_virality_model(df).

Feature Engineering: Create new columns:

title_length: The number of characters in the title.

hour_posted: The hour (0-23) extracted from the time column.

sentiment: Use the existing sentiment_score.

Target: Create a column is_viral: 1 if score > 100, else 0.

Training:

X = ['title_length', 'hour_posted', 'sentiment']

y = ['is_viral']

Train a RandomForestClassifier on this data.

Return: The trained model object."

Task 7.2: The Prediction UI

Action: Add the "Oracle" section to the sidebar in app.py.
Agent Prompt:

"Update app.py to include a prediction widget.

Train: After fetching data, call model = train_virality_model(df) (cache this using @st.cache_resource if possible).

Sidebar UI: Add an expander or section called '🔮 The Oracle'.

Input: user_title = st.text_input('Test a Headline').

Input: user_hour = st.slider('Hour of Posting', 0, 23, 12).

Prediction Logic:

When the user types a title:

Calculate its title_length.

Calculate its sentiment (re-use your analyze_sentiment logic).

Create a dataframe/array with [length, hour, sentiment].

Run model.predict_proba() to get the probability of class 1 (Viral).

Display: Show the probability as a percentage.

If > 50%: "🔥 Viral Potential!"

If < 50%: "❄️ Likely to Flop"."

✅ Success Criteria

Run streamlit run app.py.

Fetch at least 50 stories (use the slider) to give the model enough data.

Go to the Sidebar.

Type "My Cat Ate My Homework" vs. "Google Releases New AI Quantum Chip".

Win Condition:

The "Quantum Chip" title should likely have a higher probability (assuming tech words + length correlate with upvotes in your current dataset).

Changing the "Hour" slider should slightly change the probability.

⚠️ "Cold Start" Warning

If you only fetch 10 stories, the model will be terrible (it might just guess 0% for everything).

Pro Tip: For this phase, crank your slider up to 100+ stories to make the "Oracle" smarter.