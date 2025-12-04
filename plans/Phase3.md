🎨 Phase 3: The Dashboard (Streamlit)

Goal: Build a modern, interactive web dashboard to visualize your data.
Output: A file named app.py that launches a local website.
Prerequisites: You must have Phase 2 working (your data_loader.py has fetch_hn_data, analyze_sentiment, and get_topics).

📝 Context

Now that we have the "Brain" (logic), we need the "Face" (UI). We will use Streamlit, a library that turns Python scripts into websites without needing HTML or CSS.

We will build this in two layers:

The Skeleton: The layout, sidebar, and buttons.

The Visuals: The charts and metrics.

🤖 Step-by-Step AI Prompts

Task 3.1: The Skeleton

Action: Create a new file named app.py in your project folder.
Agent Prompt: (Paste into Cursor/ChatGPT)

"Create a file app.py for a Streamlit dashboard.

Imports: Import streamlit as st, plotly.express as px, and my local functions: from data_loader import fetch_hn_data, analyze_sentiment, get_topics.

Config: Set st.set_page_config(page_title='Tech Pulse', layout='wide').

Sidebar:

Add a Title 'Tech Pulse ⚡'.

Add a Slider: 'Number of Stories' (min=10, max=100, default=30).

Add a Button: 'Refresh Data'.

Main Area:

If the button is clicked (or if data is already in st.session_state):

Show a st.spinner('Fetching data...').

Call fetch_hn_data using the slider value.

Call analyze_sentiment on the result.

Call get_topics on the result.

Save the final dataframe to st.session_state['data'] so it persists.

If data exists in st.session_state, display 'Data Loaded!' and show the dataframe using st.dataframe()."

Task 3.2: The Visuals (Metrics & Charts)

Action: Update app.py to replace the simple "Data Loaded!" text with actual insights.
Agent Prompt:

"Update the display logic in app.py to show a dashboard layout instead of just the dataframe.

Metrics Row: Use st.columns(3).

Metric 1: 'Vibe Score' -> Average of sentiment_score. (Color green if > 0.05, red if < -0.05).

Metric 2: 'Total Comments' -> Sum of descendants.

Metric 3: 'Top Trend' -> The most frequent keyword in topic_keyword.

Charts Row: Use st.columns(2).

Chart 1 (Scatter): Use px.scatter with x='time', y='score', color='sentiment_label', hover_data=['title']. Title: 'Story Impact over Time'.

Chart 2 (Bar): Use px.bar to show the count of stories per topic_keyword (Top 7 topics). Title: 'Trending Topics'.

Raw Data: Display the dataframe at the bottom inside an st.expander('View Raw Data')."

✅ Success Criteria

Open your terminal in VS Code (ensure venv is active).

Run the app:

streamlit run app.py


Win Condition:

A browser tab opens automatically (http://localhost:8501).

You see a sidebar with a slider.

Clicking "Refresh Data" shows a spinner, then populates 3 big numbers, 2 charts, and a data table.

🐛 Troubleshooting

Error: ModuleNotFoundError

Fix: Ensure you ran pip install streamlit plotly in Phase 1.

Error: AttributeError: module 'data_loader' has no attribute...

Fix: Ensure your data_loader.py file is saved and actually contains the functions from Phase 1 and 2.

App keeps refreshing/reloading:

Fix: Streamlit reruns the whole script on every interaction. This is why we asked the AI to use st.session_state—to prevent re-fetching data unnecessarily.