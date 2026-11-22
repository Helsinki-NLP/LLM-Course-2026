# Let's run Gemma on CPU, if GPU is not available
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(model_name:str = "google/gemma-2b-it"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_with_chat(tokenizer, query):
    chat = [
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt") #tokenizer(prompt, return_tensors="pt")

    return inputs, prompt

def load_gemma(mobel_name:str = "google/gemma-2b-it"):
    llm_model = AutoModelForCausalLM.from_pretrained(
        mobel_name,
        torch_dtype=torch.bfloat16
    )
    return llm_model

def generate_answer(gemma_model, input_ids, tokenizer, prompt):
    outputs = gemma_model.generate(input_ids, max_new_tokens=256)
    outputs_decoded = tokenizer.decode(outputs[0])
    return outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')

# rag
def rag_prompt_formatter(tokenizer, query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: Who is Max Irwin?
Answer: Max is the CEO of Max.io, formerly he worked at OpenSource Connections delivering search improvements and running trainings.
\nExample 2:
Query: What is SolrCloud?
Answer: SolrCloud is a distributed search engine designed for improving the performance of full-text search over large datasets. It is built on top of Apache Solr, a powerful open-source search engine that provides functionality such as full-text search, faceted search, and more.
\nExample 3:
Query: What is a knowledge graph?
Answer: An instantiation of an Ontology that also contains the things that are related.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def tokenize_with_rag_prompt(tokenizer, query:str, context_items: list[dict]):
    prompt = rag_prompt_formatter(tokenizer, query, context_items)
    inputs = tokenizer.encode(prompt, add_special_tokens=False,
                              return_tensors="pt")
    return inputs, prompt