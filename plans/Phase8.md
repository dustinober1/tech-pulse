📰 Phase 8: The Executive Briefing (Auto-Report)

Goal: Generate a downloadable PDF report summarizing the current tech trends.
Output: A "Download Briefing" button in your dashboard that generates a .pdf file.
Prerequisites: You must have Phase 3 working.

📝 Context

Dashboards are great, but sometimes people just want a file to email to their boss.
We will use:

OpenAI (Optional): To generate a written summary of the trends (e.g., "AI is dominating the news today...").

FPDF: A Python library to draw text onto a PDF page.

Note: If you do not have an OpenAI key, the instructions below include a "Rule-Based" fallback where the code simply fills in a template without using an LLM.

🤖 Step-by-Step AI Prompts

Task 8.1: The PDF Generator

Action: Update data_loader.py to create the PDF.
Note: Run pip install fpdf openai first.
Agent Prompt: (Paste into Cursor/ChatGPT)

"I need a function to generate a PDF report.

Imports: Add from fpdf import FPDF and from datetime import datetime.

Function: Create generate_pdf_report(df).

Setup: Initialize pdf = FPDF(), add a page, set font to Arial 12.

Header: Add a title 'Tech Pulse: Daily Briefing' and the current date.

Stats Section: Add lines for 'Average Sentiment: X' and 'Top Topic: Y'.

Top Stories: Loop through the top 5 rows of the dataframe and print the Title and URL for each.

Output: Return the PDF as a byte string using pdf.output(dest='S').encode('latin-1').

Optional Extension: If you can, add a dummy function generate_llm_summary(titles_list) that returns a string 'AI is trending today.' and place that text at the top of the PDF."

Task 8.2: The Download Button

Action: Add the download capability to app.py.
Agent Prompt:

"Update app.py to include a download button.

Sidebar: Under the 'Refresh' button, add a 'Download Report 📄' button (or use st.download_button directly).

Logic:

Use st.download_button.

Label: 'Download PDF Briefing'.

Data: Call generate_pdf_report(st.session_state['data']).

File Name: tech_pulse_report.pdf.

Mime Type: application/pdf.

Placement: Put this in the sidebar so it's always accessible."

✅ Success Criteria

Run streamlit run app.py.

Click the "Download PDF Briefing" button in the sidebar.

Win Condition:

A file named tech_pulse_report.pdf downloads to your computer.

Open it. It should look like a simple, professional memo with today's date, stats, and the top 5 links from Hacker News.

⚠️ encoding issues

Unicode Error: fpdf sometimes crashes on emojis or fancy characters in titles.

Pro Tip: If the AI code crashes on "latin-1" encoding errors, ask the AI: "Fix the FPDF encoding error by sanitizing the text to remove non-latin characters."