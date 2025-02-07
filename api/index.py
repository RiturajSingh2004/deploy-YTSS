# api/index.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
from typing import List, Dict
import numpy as np

app = Flask(__name__)
CORS(app)

def get_video_id(url: str) -> str:
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list
    except Exception as e:
        return str(e)

def smart_chunk_transcript(transcript_list: List[Dict], target_chunk_size: int = 500) -> List[str]:
    """
    Intelligently chunk transcript based on sentence boundaries and length
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    # First, combine transcript entries into sentences
    sentences = []
    current_sentence = []
    
    for entry in transcript_list:
        text = entry['text']
        current_sentence.append(text)
        
        # Check for sentence endings
        if text.rstrip().endswith(('.', '!', '?', '..."', '"', '"')):
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Add any remaining text as a sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    # Then group sentences into chunks
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > target_chunk_size:
            if current_chunk:
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
    Process chunks efficiently with memory management
    """
    summaries = []
    
    for chunk in chunks:
        # Clear GPU memory if using CUDA
        if 'cuda' in str(summarizer.device):
            import torch
            torch.cuda.empty_cache()
            
        # Generate summary with optimized parameters
        summary = summarizer(
            chunk,
            max_length=min(len(chunk.split()) // 2, 150),  # Dynamic max length
            min_length=30,
            do_sample=False,
            num_beams=2  # Reduce beam size for memory efficiency
        )
        
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

def get_optimal_chunk_count(total_length: int) -> int:
    """
    Determine optimal number of chunks based on text length
    """
    if total_length < 1000:
        return 1
    elif total_length < 3000:
        return 2
    else:
        return 3

@app.route('/')
def home():
    return jsonify({"status": "healthy"})

@app.route('/api/summarize', methods=['POST'])
def summarize_video():
    try:
        data = request.json
        video_url = data.get('video_url', '')
        
        video_id = get_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'})
        
        transcript_list = get_transcript(video_id)
        if isinstance(transcript_list, str):  # Error occurred
            return jsonify({'error': transcript_list})
        
        # Initialize summarizer with original model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Get total transcript length
        total_text = ' '.join(entry['text'] for entry in transcript_list)
        total_length = len(total_text.split())
        
        # Smart chunking
        chunks = smart_chunk_transcript(
            transcript_list,
            target_chunk_size=1000  # Adjusted chunk size
        )
        
        # Determine optimal number of chunks to process
        n_chunks = get_optimal_chunk_count(total_length)
        chunks = chunks[:n_chunks]
        
        # Process chunks efficiently
        final_summary = process_chunks_efficiently(chunks, summarizer)
        
        return jsonify({
            'summary': final_summary,
            'video_id': video_id
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details})

if __name__ == '__main__':
    app.run()