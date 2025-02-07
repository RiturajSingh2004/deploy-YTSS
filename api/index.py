# api/index.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
from typing import List, Dict, Optional
import numpy as np
import torch

app = Flask(__name__)
CORS(app)

def get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id_match.group(1) if video_id_match else None

def get_transcript(video_id: str) -> List[Dict] | str:
    """Get transcript from YouTube video."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return str(e)

def smart_chunk_transcript(transcript_list: List[Dict], target_chunk_size: int = 500) -> List[str]:
    """
    Intelligently chunk transcript based on sentence boundaries and length.
    
    Args:
        transcript_list: List of transcript entries
        target_chunk_size: Target size for each chunk in words
        
    Returns:
        List of chunked text strings
    """
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    
    # Combine transcript entries into sentences
    sentences: List[str] = []
    current_sentence: List[str] = []
    
    sentence_endings = ('.', '!', '?', '..."', '"', '"')
    
    for entry in transcript_list:
        text = entry['text']
        current_sentence.append(text)
        
        if text.rstrip().endswith(sentence_endings):
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Add any remaining text as a sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    # Group sentences into chunks
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > target_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_chunks_efficiently(chunks: List[str], summarizer) -> str:
    """
    Process chunks efficiently with memory management.
    
    Args:
        chunks: List of text chunks to summarize
        summarizer: Hugging Face summarization pipeline
        
    Returns:
        Combined summary string
    """
    summaries: List[str] = []
    
    for chunk in chunks:
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Generate summary with optimized parameters
        summary = summarizer(
            chunk,
            max_length=min(len(chunk.split()) // 2, 150),
            min_length=30,
            do_sample=False,
            num_beams=2
        )
        
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

def get_optimal_chunk_count(total_length: int) -> int:
    """
    Determine optimal number of chunks based on text length.
    
    Args:
        total_length: Total length of text in words
        
    Returns:
        Optimal number of chunks
    """
    if total_length < 1000:
        return 1
    elif total_length < 3000:
        return 2
    else:
        return 3

@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/api/summarize', methods=['POST'])
def summarize_video():
    """
    Endpoint to summarize YouTube video transcripts.
    
    Expected JSON payload: {"video_url": "youtube_url"}
    Returns: JSON with summary and video_id, or error message
    """
    try:
        data = request.json
        video_url = data.get('video_url', '')
        
        video_id = get_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        transcript_list = get_transcript(video_id)
        if isinstance(transcript_list, str):  # Error occurred
            return jsonify({'error': transcript_list}), 400
        
        # Initialize summarizer with modern model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=device
        )
        
        # Calculate total transcript length
        total_text = ' '.join(entry['text'] for entry in transcript_list)
        total_length = len(total_text.split())
        
        # Smart chunking with adjusted size
        chunks = smart_chunk_transcript(
            transcript_list,
            target_chunk_size=1000
        )
        
        # Process optimal number of chunks
        n_chunks = get_optimal_chunk_count(total_length)
        chunks = chunks[:n_chunks]
        
        # Generate final summary
        final_summary = process_chunks_efficiently(chunks, summarizer)
        
        return jsonify({
            'summary': final_summary,
            'video_id': video_id
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'error': str(e), 
            'details': error_details
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
