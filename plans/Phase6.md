🔎 Phase 6: The Semantic Brain (Vector Search)

Goal: Implement AI-powered search that understands meaning, not just keywords.
Output: An updated app.py with a search bar that can find "AI" posts even if the title only says "LLM".
Prerequisites: You must have Phase 3 working.

📝 Context

Standard search checks if word A matches word B.
Semantic Search converts text into a list of numbers (a vector) representing its meaning.

"Dog" vector: [0.1, 0.9, 0.3]

"Puppy" vector: [0.12, 0.88, 0.31] (Mathematically close!)

"Car" vector: [0.9, 0.1, 0.0] (Far away)

We will use ChromaDB (a vector database) and Sentence Transformers to build this.

🤖 Step-by-Step AI Prompts

Task 6.1: The Vector Backend

Action: We need to update data_loader.py to handle embeddings.
Note: This requires new libraries. Run pip install chromadb sentence-transformers first.
Agent Prompt: (Paste into Cursor/ChatGPT)

"I need to add semantic search capabilities to data_loader.py.

Imports: Add import chromadb and from sentence_transformers import SentenceTransformer.

Setup:

Load the model: embedding_model = SentenceTransformer('all-MiniLM-L6-v2').

Initialize ChromaDB client: chroma_client = chromadb.Client().

Function: Create setup_vector_db(df).

Take the dataframe as input.

Convert the 'title' column to a list of strings.

Generate embeddings: embeddings = embedding_model.encode(titles_list).

Create a collection named 'hn_stories'.

Add the documents (titles), embeddings, and metadatas (original URL, score) to the collection.

Return the collection object."

Task 6.2: The Search UI

Action: Add the search bar to app.py.
Agent Prompt:

"Update app.py to include a semantic search section.

Initialize DB: After fetching data, call collection = setup_vector_db(df) (cache this if possible so we don't rebuild it every second).

UI Element: Add a text input: query = st.text_input('Ask the Hivemind (Semantic Search)').

Search Logic:

If query is not empty:

Embed the query: query_embedding = embedding_model.encode([query]).

Query the collection: results = collection.query(query_embeddings=query_embedding, n_results=5).

Display the results in a nice list or table, showing the Title and the Distance (similarity score).

Explain to the user that this search finds related concepts, not just exact words."

✅ Success Criteria

Run streamlit run app.py.

Type "Cryptocurrency" into the search bar.

Win Condition:

The app returns posts about "Bitcoin", "Blockchain", or "Wallet" even if the word "Cryptocurrency" is not in the title.

This proves the AI understands the context of the news.

⚠️ Performance Note

First Run: sentence-transformers will download a model (approx 80MB). The first time you run the search, it might lag for 5-10 seconds.

Memory: ChromaDB and Transformers require a bit more RAM than a standard script. If your computer is old, it might stutter slightly.