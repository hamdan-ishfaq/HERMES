"""
YouTube Loader — fetches transcripts and ingests into Hermes retriever.

Uses youtube-transcript-api (no API key needed).
Chunks transcript into time-stamped segments for better citations.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from src.rag.retriever import HermesRetriever


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # Assume it's already a video ID
    return url_or_id


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript segments from YouTube.
    Returns list of {text, start, duration}.
    """
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return transcript


def fetch_video_title(video_id: str) -> str | None:
    """Best-effort video title lookup via yt-dlp (returns None on failure)."""
    try:
        from yt_dlp import YoutubeDL
        opts = {"quiet": True, "skip_download": True, "no_warnings": True}
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            return info.get("title")
    except Exception:
        return None


def segments_to_chunks(segments: list[dict], chunk_seconds: int = 120) -> list[dict]:
    """
    Group transcript segments into ~2 minute chunks.
    Each chunk has a timestamp for citation.
    """
    chunks = []
    current_text = []
    current_start = 0
    current_duration = 0

    for seg in segments:
        current_text.append(seg.text)
        current_duration += seg.duration

        if current_duration >= chunk_seconds:
            chunks.append({
                "text": " ".join(current_text),
                "start_seconds": int(current_start if current_start else 0),
                "timestamp": _format_timestamp(int(current_start)),
            })
            current_text = []
            current_start = seg.start + seg.duration
            current_duration = 0

    # Remaining
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_seconds": int(current_start if current_start else 0),
            "timestamp": _format_timestamp(int(current_start)),
        })

    return chunks


def _format_timestamp(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def ingest_youtube(url_or_id: str, retriever: HermesRetriever) -> dict:
    """Full pipeline: YouTube URL → transcript → chunk → embed → store."""
    video_id = extract_video_id(url_or_id)

    print(f"\n{'='*50}")
    print(f"Ingesting YouTube: {video_id}")
    print(f"{'='*50}")

    try:
        segments = list(fetch_transcript(video_id))
    except Exception as e:
        print(f"❌ Transcript fetch failed: {e}")
        return {"video_id": video_id, "status": "failed", "error": str(e)}

    chunks = segments_to_chunks(segments, chunk_seconds=120)
    print(f"Transcript → {len(segments)} segments → {len(chunks)} chunks")

    title = fetch_video_title(video_id)

    total_parents = 0
    total_children = 0

    for chunk in chunks:
        start = chunk["start_seconds"]
        stats = retriever.ingest(
            text=chunk["text"],
            metadata={
                "source": f"youtube:{video_id}",
                "title": title,
                # Deep-link to the exact moment for this chunk.
                "url": f"https://www.youtube.com/watch?v={video_id}&t={start}s",
                "page_num": None,
                "timestamp": chunk["timestamp"],
                "start_seconds": start,
                "type": "youtube",
            }
        )
        total_parents += stats["parents"]
        total_children += stats["children"]

    result = {
        "video_id": video_id,
        "status": "ok",
        "transcript_segments": len(segments),
        "transcript_chunks": len(chunks),
        "total_parents": total_parents,
        "total_children": total_children,
    }
    print(f"Done: {result}")
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    retriever = HermesRetriever(use_cache=False, use_reranker=True)

    # 3Blue1Brown — "But what is a neural network?" (has transcript, ~19 min)
    test_video = "aircAruvnKk"
    stats = ingest_youtube(test_video, retriever)

    print("\n--- Test Query ---")
    results = retriever.query("What is a neural network?", top_k=2)
    for i, r in enumerate(results):
        ts = r["metadata"].get("timestamp", "")
        print(f"\nResult {i+1} (score: {r.get('reranker_score', r['score']):.3f}):")
        print(f"  Source: {r['metadata'].get('source')} @ {ts}")
        print(f"  {r['context'][:150]}...")
