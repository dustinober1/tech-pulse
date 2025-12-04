💾 Phase 9: The Time Machine (Historical Data)

Goal: Save every fetched story to a database so you can analyze long-term trends.
Output: An hn_history.db file and a "History" tab in your dashboard.
Prerequisites: You must have Phase 3 working.

📝 Context

Right now, your app has "Amnesia." It forgets everything the moment you close the terminal.
We will use SQLite (which comes built-in with Python) to store every story we fetch.
This enables two new features:

No Duplicates: We won't re-analyze the same story twice.

Trend Analysis: We can ask questions like "What was the top story last Tuesday?"

🤖 Step-by-Step AI Prompts

Task 9.1: The Database Manager

Action: Update data_loader.py to handle SQL storage.
Agent Prompt: (Paste into Cursor/ChatGPT)

"I need to add persistence using SQLite.

Imports: Add import sqlite3.

Init Function: Create init_db():

Connect to hn_history.db.

Create a table stories if it doesn't exist.

Columns: id (Primary Key), title, score, sentiment_score, topic_keyword, timestamp.

Save Function: Create save_stories(df):

Loop through the dataframe.

Insert each row into the stories table.

Use INSERT OR IGNORE (or check for duplicates) based on the story ID so we don't duplicate data.

Commit the changes.

Load Function: Create load_history():

Query the database: SELECT * FROM stories ORDER BY timestamp DESC LIMIT 1000.

Return this as a Pandas DataFrame."

Task 9.2: The History UI

Action: Add a tab to app.py to view the saved data.
Agent Prompt:

"Update app.py to include a History view.

Tabs: Wrap the main content in tab1, tab2 = st.tabs(['Live Dashboard', 'Time Machine']).

Move all current dashboard logic into tab1.

History Logic (Tab 2):

Call df_history = load_history().

Display a line chart: px.line(df_history, x='timestamp', y='sentiment_score', title='Sentiment Trend Over Time').

Display a dataframe of the top 50 highest-scoring posts of all time (from the DB).

Integration: Ensure that every time I click 'Refresh' in the sidebar, save_stories(new_df) is called automatically."

✅ Success Criteria

Run streamlit run app.py.

Click "Refresh Data" a few times.

Switch to the "Time Machine" tab.

Win Condition:

You see a table with more rows than the Live Dashboard (because it accumulates history).

You see a file hn_history.db appear in your VS Code file explorer.

⚠️ Database Locking

Issue: SQLite only allows one write at a time. If you refresh too fast, you might get a "Database Locked" error.

Fix: This is fine for a portfolio project. In a real app, we'd use PostgreSQL, but SQLite is perfect for single-user apps.