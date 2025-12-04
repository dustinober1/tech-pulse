🐳 Phase 10: The Container Ship (DevOps)

Goal: Package your app into a Docker container so it runs on any computer, anywhere.
Output: A Dockerfile and a running container of your app.
Prerequisites: You must have Phase 3 working and Docker Desktop installed.

📝 Context

"It works on my machine" is the most dangerous phrase in software.
Docker fixes this by packaging your OS, Python version, and libraries into a standard box (Container). If it runs in Docker, it runs on AWS, Azure, or your friend's Linux laptop.

We will create a Dockerfile—a recipe that tells Docker how to build your app.

🤖 Step-by-Step AI Prompts

Task 10.1: The Dockerfile

Action: Create a file named Dockerfile (no extension) in your project root.
Agent Prompt: (Paste into Cursor/ChatGPT)

"I need to Dockerize my Streamlit app.

Base Image: Use python:3.10-slim (lightweight version).

Setup:

Set the working directory to /app.

Copy requirements.txt into the container.

Run pip install --no-cache-dir -r requirements.txt.

App Code: Copy the rest of the current directory into /app.

Port: Expose port 8501 (Streamlit's default).

Command: Set the default command to run streamlit run app.py --server.port=8501 --server.address=0.0.0.0."

Task 10.2: The Build & Run

Action: Use the terminal to build the "ship" and set it sail.
Agent Prompt:

"Give me the specific terminal commands to:

Build the docker image and tag it as tech-pulse-app.

Run the container, mapping my local port 8501 to the container's port 8501.

Explain how I can stop it later."

✅ Success Criteria

Make sure Docker Desktop is running.

Run the build command (e.g., docker build -t tech-pulse-app .). Note: This takes time as it downloads Python.

Run the run command (e.g., docker run -p 8501:8501 tech-pulse-app).

Win Condition:

Open your browser to http://localhost:8501.

The app loads exactly as before.

Crucial Test: Stop your local Python terminal (Ctrl+C). The app should still work because it's running inside Docker, not your VS Code terminal.

⚠️ Common Pitfalls

"File not found": Make sure you are in the root folder (where app.py is) when you run the docker commands.

Database Persistence: If you restart the container, your hn_history.db will be wiped!

Advanced Fix: Ask the AI: "How do I mount a Docker Volume to save my database file outside the container?"