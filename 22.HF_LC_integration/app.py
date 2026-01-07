import streamlit as st
import validators

from urllib.parse import urlparse, parse_qs

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# ----------------------------
# Helpers (YouTube)
# ----------------------------
def extract_video_id(url: str) -> str | None:
    """
    Supports:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
    """
    try:
        u = urlparse(url)
        host = u.netloc.lower()

        if "youtu.be" in host:
            return u.path.strip("/").split("/")[0] or None

        if "youtube.com" in host:
            if u.path == "/watch":
                return parse_qs(u.query).get("v", [None])[0]
            if u.path.startswith("/shorts/"):
                return u.path.split("/shorts/")[1].split("/")[0] or None

        return None
    except Exception:
        return None


def load_youtube_docs(url: str, preferred_langs=("en",)):
    """
    Fetch transcript via youtube_transcript_api (more stable than pytube).
    Returns list[Document].
    """
    vid = extract_video_id(url)
    if not vid:
        raise ValueError("Could not extract a YouTube video ID from this URL.")

    # Try preferred languages first, then fall back
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=list(preferred_langs))
    except (TranscriptsDisabled, NoTranscriptFound):
        # try without forcing languages (let API decide)
        transcript = YouTubeTranscriptApi.get_transcript(vid)

    text = " ".join([t["text"] for t in transcript]).strip()
    if not text:
        raise ValueError("Transcript was found but is empty.")

    return [Document(page_content=text, metadata={"source": url, "video_id": vid})]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Summarize YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain + Hugging Face: Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

with st.sidebar:
    hf_api_key = st.text_input("Hugging Face API Token", value="", type="password")
    # Choose a model that is more likely to work on HF Inference without provider routing
    model_id = st.selectbox(
        "Model",
        options=[
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
        index=0,
    )
    preferred_lang = st.text_input("YouTube transcript language (optional)", value="en")

generic_url = st.text_input("URL", label_visibility="collapsed")
summarize_btn = st.button("Summarize the Content from YouTube or Website")

# ----------------------------
# Build LLM
# ----------------------------
def build_llm(hf_token: str, repo_id: str):
    # IMPORTANT:
    # - use huggingfacehub_api_token (NOT token=)
    # - conversational works for chat-style endpoints; ChatHuggingFace wraps it safely
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=350,
        temperature=0.7,
    )
    return ChatHuggingFace(llm=endpoint)


prompt_template = """
Provide a clear summary of the following content in about 300 words.

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ----------------------------
# Run
# ----------------------------
if summarize_btn:
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the Hugging Face token and a URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or a webpage).")
        st.stop()

    try:
        llm = build_llm(hf_api_key, model_id)

        with st.spinner("Loading content..."):
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                # Use transcript API (more reliable than YoutubeLoader/pytube)
                langs = (preferred_lang.strip(),) if preferred_lang.strip() else ("en",)
                docs = load_youtube_docs(generic_url, preferred_langs=langs)
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                docs = loader.load()

        with st.spinner("Summarizing..."):
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            result = chain.invoke({"input_documents": docs})
            output_summary = result["output_text"]

        st.success(output_summary)

    except TranscriptsDisabled:
        st.error("This YouTube video has transcripts/subtitles disabled.")
    except NoTranscriptFound:
        st.error("No transcript was found for this YouTube video (try a different video).")
    except Exception as e:
        st.exception(e)
