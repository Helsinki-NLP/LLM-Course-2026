from stqdm import stqdm
import re

def sentencize(pages_and_texts: list[dict], nlp):
    for item in stqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])


# chunking, i.e. grouping sentences into chunks of text
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Perform sentence chunking
def chunk(pages_and_texts: list[dict]):
    # Loop through pages and texts and split sentences into chunks
    for item in stqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])


# Improved chunking with overlapping windows for better precision
def split_list_overlapping(input_list: list,
                          chunk_size: int,
                          overlap: int) -> list[list[str]]:
    """
    Splits the input_list into overlapping sublists.
    
    Args:
        input_list: List of sentences to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
    
    Example:
        With chunk_size=5, overlap=2, a list of 10 sentences would create:
        [[0-4], [3-7], [6-9]]
    """
    if len(input_list) <= chunk_size:
        return [input_list]
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(input_list), step):
        chunk = input_list[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            # Stop if we've covered all sentences
            if i + chunk_size >= len(input_list):
                break
    return chunks


def detect_section_header(sentence: str) -> bool:
    """
    Detects if a sentence looks like a section header.
    Section headers are typically:
    - Short sentences
    - May contain numbers (like "15.2.1")
    - Often start with capital letters
    - May be all caps or have specific patterns
    """
    sentence = sentence.strip()
    if not sentence:
        return False
    
    # Pattern 1: Numbered sections like "15.2.1 Retrieval augmented generation"
    # This is the most reliable pattern - matches numbered sections followed by title case text
    if re.match(r'^\d+\.\d+(\.\d+)?\s+[A-Z]', sentence):
        return True
    
    # Pattern 2: Numbered sections with optional sub-numbering
    # Matches patterns like "15.2.1", "15.2", etc. at the start
    if re.match(r'^\d+\.\d+(\.\d+)*\s', sentence):
        return True
    
    # Pattern 3: Short sentences that might be headers (less than 100 chars, title case, no period)
    # Headers typically don't end with periods
    if len(sentence) < 100 and sentence[0].isupper() and not sentence.endswith('.'):
        # Check if it looks like a title (mostly title case words)
        words = sentence.split()
        if len(words) > 0 and len(words) < 15:
            # Most words should start with capital letters (title case)
            title_case_words = sum(1 for w in words if w and w[0].isupper())
            if title_case_words >= len(words) * 0.6:  # At least 60% title case
                return True
    
    # Pattern 4: All caps short sentences (typical for section headers)
    if len(sentence) < 80 and sentence.isupper() and len(sentence.split()) < 10:
        return True
    
    return False


def chunk_improved(pages_and_texts: list[dict], 
                   chunk_size: int = 5,
                   overlap: int = 2,
                   min_chunk_size: int = 2):
    """
    Improved chunking strategy with overlapping windows and semantic awareness.
    
    Args:
        pages_and_texts: List of page dictionaries with sentences
        chunk_size: Target number of sentences per chunk (default: 5 for better precision)
        overlap: Number of sentences to overlap between chunks (default: 2)
        min_chunk_size: Minimum chunk size to keep (default: 2)
    """
    for item in stqdm(pages_and_texts):
        sentences = item["sentences"]
        if not sentences:
            item["sentence_chunks"] = []
            item["num_chunks"] = 0
            continue
        
        chunks = []
        
        # Use overlapping window approach
        overlapping_chunks = split_list_overlapping(sentences, chunk_size, overlap)
        
        # Also create smaller, focused chunks around section headers
        # This helps capture definitions and key concepts more precisely
        for i, sentence in enumerate(sentences):
            if detect_section_header(sentence):
                # Create multiple focused chunks for better coverage:
                # 1. Header + 1-2 sentences (for definitions)
                # 2. Header + 3-5 sentences (for longer explanations)
                # This ensures we capture both concise definitions and longer explanations
                
                # Very focused chunk: header + 1-2 sentences (great for definitions)
                if i + 2 <= len(sentences):
                    focused_short = sentences[i:min(i + 3, len(sentences))]
                    if len(focused_short) >= 2:
                        chunks.append(focused_short)
                
                # Medium focused chunk: header + 3-5 sentences
                focused_medium = sentences[i:min(i + chunk_size, len(sentences))]
                if len(focused_medium) >= min_chunk_size:
                    chunks.append(focused_medium)
        
        # Combine overlapping chunks with focused header chunks
        # Remove duplicates (chunks that are identical)
        all_chunks = overlapping_chunks + chunks
        
        # Deduplicate while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            chunk_tuple = tuple(chunk)
            if chunk_tuple not in seen:
                seen.add(chunk_tuple)
                unique_chunks.append(chunk)
        
        # Filter out chunks that are too small
        item["sentence_chunks"] = [c for c in unique_chunks if len(c) >= min_chunk_size]
        item["num_chunks"] = len(item["sentence_chunks"])

# Convert chunks into text elements ready for embedding
def chunks_to_text_elems(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in stqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks