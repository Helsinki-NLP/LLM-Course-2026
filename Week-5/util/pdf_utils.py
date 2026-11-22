from stqdm import stqdm
import fitz
import numpy as np

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in stqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({
                                #"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                                "page_number": page_number - 4,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts

def load_page(filename:str, page_num: int, query: str):
    # Open PDF and load target page
    doc = fitz.open(filename)
    # page = doc.load_page(5 + 41) # number of page (our doc starts page numbers on page 41)
    page = doc.load_page(238 + 4)

    # Get the image of the page
    img = page.get_pixmap(dpi=300)

    # Optional: save the image
    # img.save("output_filename.png")
    doc.close()

    # Convert the Pixmap to a numpy array
    img_array = np.frombuffer(img.samples_mv,
                              dtype=np.uint8).reshape((img.h, img.w, img.n))

    # Display the image using Matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 10))
    plt.imshow(img_array)
    plt.title(f"Query: '{query}' | Most relevant page:")
    plt.axis('off')  # Turn off axis
    plt.show()
