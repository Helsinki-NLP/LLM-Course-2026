import streamlit as st
import spacy
from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, chunk, chunks_to_text_elems
import pandas as pd
from util.generator_utils import load_tokenizer, tokenize_with_chat, tokenize_with_rag_prompt, load_gemma, generate_answer
from util.session_utils import SESSION_VARS, put_to_session, get_from_session, print_session
from util.vector_search_utils import retrieve_relevant_resources

# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

min_token_length = 30
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

    # Gemma
    model = "google/gemma-2b-it"
    gemma_model = load_gemma(model)
    tokenizer = load_tokenizer(model)

    put_to_session(st, SESSION_VARS.MODEL, gemma_model)
    put_to_session(st, SESSION_VARS.TOKENIZER, tokenizer)

    st.write("Done")

    put_to_session(st, SESSION_VARS.LOADED_MODELS, True)
else:
    st.write("Models were already loaded")

print_session(st)

st.title('PDF RAG (Retrieval Augmented Generation) Demo')
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

if uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
    if uploaded_file.name != get_from_session(st, SESSION_VARS.CUR_PDF_FILENAME):
        put_to_session(st, SESSION_VARS.PROCESSED_DATA, None)
        put_to_session(st, SESSION_VARS.CUR_PDF_FILENAME, uploaded_file.name)

    # let's process the file, if it is a new one
    if not get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        with st.expander("Preprocessing"):
            st.write("Reading pdf")
            pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)
            # print(pages_and_texts[:2])
            # extract sentences
            st.write("Extracting sentences")
            sentencize(pages_and_texts, get_from_session(st, SESSION_VARS.NLP))
            # chunk
            st.write("Chunking")
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

    if get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        st.write("Vector Search")
        st.write("Loading embeddings to tensor")
        tensor, pages_and_chunks = embeddings_to_tensor(get_from_session(st, SESSION_VARS.EMBEDDINGS_FILENAME))
        scores, indices = retrieve_relevant_resources(query, tensor, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU), st)
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]
        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()  # return score back to CPU
        st.write(f"Query: {query}")
        with st.expander("Results"):
            # Loop through zipped together scores and indicies
            for score, index in zip(scores, indices):
                st.write(f"Score: {score:.4f}")
                # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
                st.write(pages_and_chunks[index]["sentence_chunk"])
                # Print the page number too so we can reference the textbook further and check the results
                st.write(f"Page number: {pages_and_chunks[index]['page_number']}")

        st.write("You selected:", gen_variant)
        with st.expander(f"Answer for query: {query}"):
            with st.spinner("Generating"):
                if gen_variant == "vanilla":
                    input_ids, prompt = tokenize_with_chat(get_from_session(st, SESSION_VARS.TOKENIZER), query)
                    answer = generate_answer(get_from_session(st, SESSION_VARS.MODEL), input_ids, get_from_session(st, SESSION_VARS.TOKENIZER), prompt)
                    st.write(answer)
                elif gen_variant == "rag":
                    input_ids, prompt = tokenize_with_rag_prompt(get_from_session(st, SESSION_VARS.TOKENIZER), query, context_items)
                    answer = generate_answer(get_from_session(st, SESSION_VARS.MODEL), input_ids, get_from_session(st, SESSION_VARS.TOKENIZER), prompt)
                    st.write(answer)
        st.success("Done!")
