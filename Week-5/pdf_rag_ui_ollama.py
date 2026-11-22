import streamlit as st
import spacy
import re
from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, chunk, chunk_improved, chunks_to_text_elems
import pandas as pd
from util.session_utils import SESSION_VARS, put_to_session, get_from_session, print_session
from util.vector_search_utils import retrieve_relevant_resources
import ollama

# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

min_token_length = 30

def ollama_model_id(model) -> str | None:
    """Installed model id from either the current client or older dict responses.

    ollama-python 0.4+ returns Pydantic models whose id is ``model``
    (for example ``llama3:latest``). Earlier clients returned dicts with ``name``.
    """
    if isinstance(model, dict):
        return model.get("model") or model.get("name")
    return getattr(model, "model", None) or getattr(model, "name", None)

def ollama_has_model(requested: str, installed: list[str]) -> bool:
    if requested in installed:
        return True
    # Untagged names resolve to the :latest tag in the local model list.
    return ":" not in requested and f"{requested}:latest" in installed

# Ollama model name - user can change this
OLLAMA_MODEL = st.sidebar.text_input("Ollama Model Name", value="llama3", help="Name of the Ollama model to use (e.g., llama3, mistral, gemma:2b)")

# Chunking strategy selection
CHUNKING_STRATEGY = st.sidebar.selectbox(
    "Chunking Strategy",
    ("improved", "original"),
    help="Improved: Uses overlapping windows and semantic-aware chunking for better precision. Original: Fixed-size non-overlapping chunks."
)

# Number of results to return
N_RESULTS = st.sidebar.slider(
    "Number of Results",
    min_value=3,
    max_value=20,
    value=10,
    help="Number of top results to return from vector search"
)

# Hybrid search option
USE_HYBRID = st.sidebar.checkbox(
    "Use Hybrid Search",
    value=True,
    help="Combine vector search with keyword matching to boost definition chunks"
)

# Generation parameters
MAX_TOKENS = st.sidebar.slider(
    "Max Tokens",
    min_value=128,
    max_value=2048,
    value=512,
    step=128,
    help="Maximum number of tokens to generate (higher = longer answers, but slower)"
)

st.write("Initializing models")

if not get_from_session(st, SESSION_VARS.LOADED_MODELS):
    nlp = spacy.load("en_core_web_sm") #English()

    # uncomment this command to print the file location of the Spacy model
    # st.write(nlp._path)

    # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
    nlp.add_pipe("sentencizer")
    put_to_session(st, SESSION_VARS.NLP, nlp)

    embedding_model_cpu = SentenceTransformer(model_name_or_path="/Users/dmitrykan/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130",
                                          device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    put_to_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU, embedding_model_cpu)

    # Ollama client - no need to load model, it's handled by Ollama service
    # Just verify the model is available
    try:
        # Check if model exists
        models = ollama.list()
        model_names = [name for model in models["models"] if (name := ollama_model_id(model))]
        if not ollama_has_model(OLLAMA_MODEL, model_names):
            st.warning(f"Model '{OLLAMA_MODEL}' not found in Ollama. Available models: {', '.join(model_names)}")
            st.info(f"You can pull the model with: ollama pull {OLLAMA_MODEL}")
        else:
            st.success(f"Using Ollama model: {OLLAMA_MODEL}")
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        st.info("Make sure Ollama is running. You can start it with: ollama serve")

    put_to_session(st, SESSION_VARS.MODEL, OLLAMA_MODEL)  # Store model name instead of model object

    st.write("Done")

    put_to_session(st, SESSION_VARS.LOADED_MODELS, True)
else:
    st.write("Models were already loaded")

print_session(st)

st.title('PDF RAG (Retrieval Augmented Generation) Demo - Ollama Version')
query = st.text_input("Type your query here", "What is signal boosting?")
gen_variant = st.selectbox(
    "Select vanilla LLM or Retrieval Augmented LLM",
    ("vanilla", "rag")
)

uploaded_file = st.file_uploader(
    label="Upload a pdf",
    help="Upload a pdf file to chat to it with RAG",
    type='pdf'
)

button_clicked = st.button("Generate")

def format_vanilla_prompt(query: str) -> str:
    """Format a simple query prompt for vanilla LLM."""
    return query

def format_rag_prompt(query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    Adapted for Ollama which doesn't need chat templates.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: Who is Max Irwin?
Answer: Max is the CEO of Max.io, formerly he worked at OpenSource Connections delivering search improvements and running trainings.

Example 2:
Query: What is SolrCloud?
Answer: SolrCloud is a distributed search engine designed for improving the performance of full-text search over large datasets. It is built on top of Apache Solr, a powerful open-source search engine that provides functionality such as full-text search, faceted search, and more.

Example 3:
Query: What is a knowledge graph?
Answer: An instantiation of an Ontology that also contains the things that are related.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    prompt = base_prompt.format(context=context, query=query)
    return prompt

def generate_answer_ollama(model_name: str, prompt: str, max_tokens: int = 512) -> str:
    """Generate answer using Ollama API."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'num_predict': max_tokens,  # max_new_tokens equivalent
                'temperature': 0.7,  # Balance between creativity and consistency
            }
        )
        answer = response['message']['content']
        
        # Check if response might be truncated (ends mid-sentence or with incomplete list)
        if answer and len(answer) > 50:  # Only check if answer is substantial
            answer_stripped = answer.rstrip()
            # Check for incomplete patterns that suggest truncation
            incomplete_patterns = [
                answer_stripped.endswith(','),  # Ends with comma
                answer_stripped.endswith(';'),  # Ends with semicolon
                bool(re.search(r'\d+\.\s*$', answer_stripped)),  # Ends with "2. " or similar (incomplete list)
                # Check if ends mid-sentence (no proper ending punctuation and ends with lowercase)
                (not answer_stripped.endswith('.') and 
                 not answer_stripped.endswith('?') and 
                 not answer_stripped.endswith('!') and
                 len(answer_stripped) > 0 and
                 answer_stripped[-1].islower() and
                 len(answer_stripped.split()) > 10),  # Has substantial content
            ]
            if any(incomplete_patterns):
                answer += "\n\n[Note: Response may be truncated. Consider increasing Max Tokens in the sidebar.]"
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

if uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
    # Check if we need to reprocess: new file or chunking strategy changed
    stored_filename = get_from_session(st, SESSION_VARS.CUR_PDF_FILENAME)
    stored_chunking = st.session_state.get("chunking_strategy", None)
    
    if uploaded_file.name != stored_filename or CHUNKING_STRATEGY != stored_chunking:
        put_to_session(st, SESSION_VARS.PROCESSED_DATA, None)
        put_to_session(st, SESSION_VARS.CUR_PDF_FILENAME, uploaded_file.name)
        st.session_state["chunking_strategy"] = CHUNKING_STRATEGY

    # Index the PDF only for RAG. Vanilla answers from the model alone.
    if gen_variant == "rag" and not get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        with st.expander("Preprocessing"):
            st.write("Reading pdf")
            pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)
            # print(pages_and_texts[:2])
            # extract sentences
            st.write("Extracting sentences")
            sentencize(pages_and_texts, get_from_session(st, SESSION_VARS.NLP))
            # chunk
            st.write(f"Chunking (strategy: {CHUNKING_STRATEGY})")
            if CHUNKING_STRATEGY == "improved":
                chunk_improved(pages_and_texts, chunk_size=5, overlap=2, min_chunk_size=2)
            else:
                chunk(pages_and_texts)
            # chunks to text elems
            pages_and_chunks = chunks_to_text_elems(pages_and_texts)
            st.write("Loading to a DataFrame")
            df = pd.DataFrame(pages_and_chunks)
            # Let's filter our DataFrame/list of dictionaries to only include chunks with over 30 tokens in length
            pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
            st.write("Embedding")
            embed_chunks(pages_and_chunks_over_min_token_len, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU))
            st.write("Saving embeddings")
            filename = save_embeddings(pages_and_chunks_over_min_token_len)

            put_to_session(st, SESSION_VARS.EMBEDDINGS_FILENAME, filename)
            put_to_session(st, SESSION_VARS.PROCESSED_DATA, True)

    st.write("You selected:", gen_variant)
    st.write(f"Query: {query}")
    model_name = OLLAMA_MODEL

    if gen_variant == "vanilla":
        st.info("Vanilla LLM: the PDF is not searched, and no passages are added to the prompt.")
        with st.expander(f"Answer for query: {query}", expanded=True):
            with st.spinner("Generating"):
                prompt = format_vanilla_prompt(query)
                answer = generate_answer_ollama(model_name, prompt, max_tokens=MAX_TOKENS)
                st.write(answer)
        st.success("Done!")
    elif get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        st.write("Vector Search")
        st.write("Loading embeddings to tensor")
        tensor, pages_and_chunks = embeddings_to_tensor(get_from_session(st, SESSION_VARS.EMBEDDINGS_FILENAME))
        scores, indices = retrieve_relevant_resources(
            query, tensor, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU), st,
            n_resources_to_return=N_RESULTS,
            pages_and_chunks=pages_and_chunks,
            use_hybrid=USE_HYBRID
        )
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]
        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()  # return score back to CPU
        with st.expander("Results"):
            # Loop through zipped together scores and indicies
            for score, index in zip(scores, indices):
                st.write(f"Score: {score:.4f}")
                # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
                st.write(pages_and_chunks[index]["sentence_chunk"])
                # Print the page number too so we can reference the textbook further and check the results
                st.write(f"Page number: {pages_and_chunks[index]['page_number']}")

        with st.expander(f"Answer for query: {query}", expanded=True):
            with st.spinner("Generating"):
                prompt = format_rag_prompt(query, context_items)
                answer = generate_answer_ollama(model_name, prompt, max_tokens=MAX_TOKENS)
                st.write(answer)
        st.success("Done!")

