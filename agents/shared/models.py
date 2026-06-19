# agents/shared/models.py
#
# One place to create LLM and embedding instances.
# Change model names or providers here and every agent picks it up.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv('/Users/nnandagopal/Desktop/personal_projects/RAG/.env')

CHAT_MODEL = "gpt-4o-mini"
STRONG_CHAT_MODEL = "gpt-4o-mini"          # used by deep research for planning/reflection
CLASSIFIER_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Module-level cache so we don't create new instances on every call
_chat: ChatOpenAI | None = None
_strong_chat: ChatOpenAI | None = None
_embeddings: OpenAIEmbeddings | None = None


def get_llm(strong: bool = False) -> ChatOpenAI:
    """Return the standard (or strong) chat model."""
    global _chat, _strong_chat
    if strong:
        if _strong_chat is None:
            _strong_chat = ChatOpenAI(model=STRONG_CHAT_MODEL)
        return _strong_chat
    if _chat is None:
        _chat = ChatOpenAI(model=CHAT_MODEL)
    return _chat


def get_classifier_llm() -> ChatOpenAI:
    """Return a fast, cheap model used only for classification tasks."""
    return ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings