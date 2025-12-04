🚀 Phase 4: Deployment (Go Live)

Goal: Publish your local dashboard to the internet using Streamlit Cloud.
Output: A live URL (e.g., https://my-tech-pulse.streamlit.app).
Prerequisites: You must have Phase 3 working locally (your app runs on localhost:8501).

📝 Context

To run this in the cloud, Streamlit's servers need to know two things:

The Code: We will share this via GitHub.

The Tools: We will list every library we used (Pandas, Plotly, etc.) in a file called requirements.txt.

Streamlit Cloud is free and connects directly to your GitHub account. When you update your code on GitHub, your live website updates automatically.

🤖 Step-by-Step AI Prompts

Task 4.1: The Dependencies (requirements.txt)

Action: We need a strict list of libraries.
Agent Prompt: (Paste into Cursor/ChatGPT)

"Scan my app.py and data_loader.py files. Create the exact content I need for a requirements.txt file.

Rules:

Include streamlit, pandas, requests, plotly, nltk, bertopic, and scikit-learn.

Do NOT include standard libraries like os or time.

Do NOT include venv or system paths.
Just give me the list so I can paste it into a file."

Manual Action:

Create a file named requirements.txt in your project folder.

Paste the list provided by the AI.

Task 4.2: Git Safety (.gitignore)

Action: Prevent junk files from being uploaded.
Agent Prompt:

"Create the content for a .gitignore file for a Python Streamlit project. It should exclude:

venv/ (The virtual environment)

__pycache__/ (Compiled python files)

.env (Secrets, just in case)

.DS_Store (Mac system files)"

Manual Action: Create a file named .gitignore and paste the content.

🌍 Task 4.3: Going Live (Manual Steps)

Since the AI cannot click buttons for you, follow these exact steps:

Push to GitHub:

Go to GitHub.com and create a new public repository named tech-pulse.

In your VS Code Terminal, run:

git init
git add .
git commit -m "Initial deploy"
git branch -M main
git remote add origin [https://github.com/YOUR_USERNAME/tech-pulse.git](https://github.com/YOUR_USERNAME/tech-pulse.git)
git push -u origin main


Deploy to Streamlit:

Go to share.streamlit.io.

Sign in with GitHub.

Click "New App".

Repository: Select tech-pulse.

Main Module: app.py.

Click "Deploy".

✅ Success Criteria

Wait for the "Oven" animation to finish (it might take 2-3 minutes to install BERTopic).

Win Condition:

You get a confetti animation.

You have a URL like https://tech-pulse-app.streamlit.app.

The app works exactly like it did on your computer.

🐛 Troubleshooting

Error: ModuleNotFoundError on Cloud:

Fix: You forgot to add that specific library to requirements.txt. Add it, commit, and push again.

Error: High Memory Usage / App Crash:

Fix: BERTopic can be heavy. If the free tier crashes, update data_loader.py to use a lighter model: BERTopic(embedding_model="paraphrase-MiniLM-L3-v2").