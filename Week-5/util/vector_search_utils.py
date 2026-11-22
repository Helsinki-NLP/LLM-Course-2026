import torch
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import re

# Define helper function to print wrapped text
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                st,
                                n_resources_to_return: int=5,
                                print_time: bool=True,
                                pages_and_chunks: list[dict] = None,
                                use_hybrid: bool = False):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    
    Args:
        query: Search query
        embeddings: Tensor of embeddings for all chunks
        model: SentenceTransformer model
        st: Streamlit object for UI
        n_resources_to_return: Number of results to return
        print_time: Whether to print timing info
        pages_and_chunks: List of chunk dictionaries (needed for hybrid search)
        use_hybrid: If True, combines vector search with keyword matching
    """

    # Embed the query
    query_embedding = model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        # print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
        st.write(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    # Apply hybrid search boost if enabled
    if use_hybrid and pages_and_chunks is not None:
        # Boost chunks that contain query keywords
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove common stop words from query words for better matching
        stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'was', 'were', 'does', 'do', 'how', 'why', 'when', 'where'}
        query_words = {w for w in query_words if w not in stop_words and len(w) > 2}
        
        # Definition patterns that indicate a definition is present
        definition_patterns = [
            'commonly referred to',
            'refers to',
            'defined as',
            'is defined',
            'denotes',
            'means',
            'is a',
            'is an',
            'are',
            'workflow of using'
        ]
        
        for i, chunk_dict in enumerate(pages_and_chunks):
            chunk_text = chunk_dict.get("sentence_chunk", "").lower()
            chunk_text_original = chunk_dict.get("sentence_chunk", "")
            
            # Boost if chunk contains query words
            chunk_words = set(chunk_text.split())
            query_match_count = len(query_words.intersection(chunk_words))
            if query_match_count > 0:
                # Boost by 0.08 per matching query word (increased from 0.05)
                dot_scores[i] += query_match_count * 0.08
            
            # Check if chunk starts with a numbered section header (like "15.2.1")
            starts_with_numbered_section = re.match(r'^\d+\.\d+(\.\d+)?\s+', chunk_text_original)
            
            # Extra boost if it looks like a definition
            is_definition = any(pattern in chunk_text for pattern in definition_patterns)
            
            if is_definition:
                # Check if it also contains query terms
                if query_match_count > 0:
                    # Strong boost for definition chunks with query terms
                    boost_amount = 0.25  # Increased from 0.15
                    # Even stronger if it starts with a numbered section
                    if starts_with_numbered_section:
                        boost_amount += 0.15  # Extra boost for numbered section definitions
                    dot_scores[i] += boost_amount
                else:
                    # Still boost definition chunks slightly even without exact query match
                    dot_scores[i] += 0.08  # Increased from 0.05
            
            # Boost chunks that start with numbered sections and contain query terms
            if starts_with_numbered_section and query_match_count > 0:
                dot_scores[i] += 0.12  # Boost for numbered sections with query terms

    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """

    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")