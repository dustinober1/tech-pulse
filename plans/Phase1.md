🔌 Phase 1: The Data Pipeline (Hacker News)

Goal: Create a robust Python script that fetches raw data from Hacker News.
Output: A file named data_loader.py that prints a DataFrame of tech news.
Prerequisites: You must have completed Phase 0 (Python installed, Virtual Environment activated, requests and pandas installed).

📝 Context

Before we analyze anything, we need data. Hacker News doesn't require an API key, but it requires two steps to get data:

Get IDs: Fetch the list of top 500 story IDs.

Get Details: Loop through those IDs and fetch the JSON details for each.

We will write a script to handle this automatically.

🤖 Step-by-Step AI Prompts

Task 1.1: The Fetch Function

Action: Create a new file named data_loader.py in your project folder.
Agent Prompt: (Paste this into Cursor/ChatGPT)

"Write a Python function called fetch_hn_data(limit=30).
It should use the requests library.

Fetch IDs: GET https://hacker-news.firebaseio.com/v0/topstories.json. This returns a list of integers.

Fetch Details: Loop through the first limit IDs from that list. For each ID, GET https://hacker-news.firebaseio.com/v0/item/{id}.json.

Data Extraction: For each item, extract these fields into a dictionary:

title (str)

score (int) - The number of upvotes

descendants (int) - The comment count (handle cases where it's missing, default to 0)

time (datetime) - Convert the Unix timestamp to a readable Python datetime object immediately.

url (str) - The link to the article

Error Handling: Wrap the network requests in try/except blocks. If a single story fails to load, skip it and print a warning, but don't crash the whole script.

Return: Convert the list of dictionaries into a Pandas DataFrame and return it."

Task 1.2: Verification (The Sanity Check)

Action: We need to make sure this file runs on its own.
Agent Prompt:

"Add a standard Python entry point to the bottom of data_loader.py.

if __name__ == '__main__':
    print('Fetching data...')
    df = fetch_hn_data(limit=5)
    print(df.head())
    print(f'Successfully fetched {len(df)} stories.')



Ensure that all necessary imports (`requests`, `pandas`) are at the top of the file."


✅ Success Criteria

Open your terminal in VS Code.

Ensure your virtual environment is active (you see (venv)).

Run the script:

python data_loader.py



Win Condition: You see a table of data printed in your terminal looking something like this:

   title                          score  descendants  time        url
0  New AI Model Released...       150    45           171...      https://...
1  Why Rust is Great...           89     12           171...      https://...
...
Successfully fetched 5 stories.



🐛 Troubleshooting

Error: ModuleNotFoundError: No module named 'requests'

Fix: You forgot to install the tools. Run pip install requests pandas.

Error: ConnectionError

Fix: Check your internet. The Hacker News API might be blocked on your network.

Empty Data:

Fix: Check the API URL spelling. It must be exact.