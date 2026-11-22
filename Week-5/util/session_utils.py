from enum import Enum
import streamlit as st

class SESSION_VARS(Enum):
    LOADED_MODELS = 1,
    CUR_PDF_FILENAME = 2,
    PROCESSED_DATA = 3,
    NLP = 4,
    EMBEDDING_MODEL_CPU = 5,
    TOKENIZER = 6,
    MODEL = 7,
    EMBEDDINGS_FILENAME = 8


def put_to_session(st: st, key:SESSION_VARS, value):
    """
        Update the Streamlit session state with the given key-value pair.

        Parameters:
        st_session_state (dict): The st.session_state object.
        key (str): The key to update or add in session state.
        value: The value to assign to the key.
        """
    if key not in st.session_state:
        st.session_state[key.name] = value

# Getter function
def get_from_session(st: st, key: SESSION_VARS, default=None):
    """
    Retrieve a value from Streamlit session state by key.

    Parameters:
    st_session_state (dict): The st.session_state object.
    key (str): The key to retrieve from session state.
    default: The default value to return if the key does not exist.

    Returns:
    The value associated with the key or the default value.
    """
    return st.session_state.get(key.name, default)

def print_session(st):
    print(st.session_state)