import os, json, time, yaml, subprocess, concurrent.futures, mimetypes
from faster_whisper import WhisperModel
from google import genai
from google.genai import types


# CONFIGURATION
API_KEY = ""  # Replace with actual key
INPUT_PATH = r"sample_pitch_audio_output.mp3"
PROMPT_DIR = r"Prompts"   # Ensure this directory exists with your yaml files

model_size = "small.en"
transcribe_model = WhisperModel(model_size, device="cpu", compute_type="int8")

client = genai.Client(api_key=API_KEY)



def is_video_file(file_path):

    """
    Determines if a file is a video based on MIME type or extension.
    """

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('video'):
        return True
    
    # Fallback for common extensions if mime detection fails
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def split_audio_video(input_path):

    """Splits input video into separate audio and silent video files."""

    print(f'[FFMPEG] Splitting audio and video...')
    base_name = os.path.splitext(input_path)[0]
    output_audio = f"{base_name}_audio_output.mp3"
    output_video = f"{base_name}_video_output.mp4"

    # FFMPEG commands (suppressing verbose output for cleaner logs)
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "libmp3lame", output_audio], 
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-an", "-vf", "scale=640:360,fps=5", "-c:v", "libx264", "-preset", "veryfast", output_video], 
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # subprocess.run(["ffmpeg", "-y", "-i", input_path, "-an", "-c:v", "copy", output_video], 
    #                 check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f'[FFMPEG] Split complete.')
    return output_audio, output_video


def llm_transcribe_audio(audio_path):

    """Sends audio to Gemini for transcription."""

    print(f'[Transcribe] Starting transcription for {audio_path}...')
    start_t = time.time()

    # Load Prompt
    prompt_path = os.path.join(PROMPT_DIR, "PitchAI_prompt_transcribe.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)
    
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget = 0), # Consider setting to 0 for speed
            system_instruction=config["system_prompt"]

        ),
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
        ]
    )
    
    transcript_data = json.loads(response.text)
    
    # Save intermediate result
    base_name = os.path.splitext(audio_path)[0].replace("_audio_output", "")
    with open(f"{base_name}_transcript.json", 'w') as f:
        json.dump(transcript_data, f, indent=4)
        
    print(f'[Transcribe] Finished in {time.time() - start_t:.2f}s')
    return transcript_data


def transcribe_audio(audio_path, output_json_path="transcript.json"):
    print("Starting transcription...")
    start_time = time.time()

    segments, info = transcribe_model.transcribe(
        audio_path,
        beam_size=4,
        word_timestamps=False,
        language="en",
        task="transcribe",
        temperature=0.0,
        vad_filter=True
    )

    transcript_data = {
        "language": info.language,
        "duration": info.duration,
        "segments": [],
        "full_transcript": ""
    }

    full_text_parts = []
    for segment in segments:
        segment_dict = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }

        transcript_data["segments"].append(segment_dict)
        full_text_parts.append(segment.text.strip())

    transcript_data["full_transcript"] = " ".join(full_text_parts)

    # Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    full_transcript = {key: transcript_data[key] for key in ["full_transcript"]}

    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds.")
    print(f"Saved transcript to {output_json_path}")

    return full_transcript


def evaluate_text(transcript_data):
    """Analyzes the transcript JSON."""
    print(f'[Text Eval] Starting text analysis...')
    start_t = time.time()

    prompt_path = os.path.join(PROMPT_DIR, "PitchAI_prompt_text.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)

    text_sys_prompt = config["system_prompt"].format(
        PRODUCT_NAME="Not Provided", TARGET_PERSONA="Not Provided",
        DEAL_STAGE="Not Provided", PAIN_POINTS="Not Provided",
        USP="Not Provided", COMPETITORS="Not Provided",
        MANDATORY_TERMS="Not Provided", FORBIDDEN_TERMS="Not Provided"
    )

    # Prepare inputs
    transcript_str = json.dumps(transcript_data, indent=2)
    text_user_prompt = config["user_prompt"].format(transcript=transcript_str)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget = -1),
            system_instruction = text_sys_prompt
        ),
        contents=[
            types.Content(role="user", parts=[
                types.Part.from_text(text = text_user_prompt)
            ])
        ]
    )

    try:
        result = json.loads(response.text)
    except Exception as e:
        print(f"[Error] Failed to parse JSON response: {e}")
        result = {"error": "Invalid JSON response from LLM", "raw_text": response.text}

    print(f'[Text Eval] Finished in {time.time() - start_t:.2f}s')

    return result


def evaluate_video(video_path):
    """
    Hybrid Logic:
    - If video < 18MB: Send Inline (Fastest, ~30s latency)
    - If video >= 18MB: Upload via File API (Slower, but won't crash)
    """
    print(f'[Video Eval] Starting video analysis for {video_path}...')
    start_t = time.time()
    
    prompt_path = os.path.join(PROMPT_DIR, "PitchAI_prompt_visual.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)

    sys_prompt = config["system_prompt"]

    # 1. Check File Size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f'[Video Eval] Optimized Video Size: {file_size_mb:.2f} MB')

    # THRESHOLD: 18MB (Leaving 2MB buffer for prompt text and overhead)
    if file_size_mb < 18.0:
        # --- METHOD A: INLINE (FAST) ---
        print(f'[Video Eval] Size is under limit. Using INLINE mode (Fast)...')
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget = 0),
                system_instruction=sys_prompt
            ),
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
            ]
        )
    else:
        # --- METHOD B: FILE API (SAFE) ---
        print(f'[Video Eval] Size is over limit. Using FILE API mode (Safe)...')
        video_file = client.files.upload(file=video_path)
        
        # Fast Polling
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = client.files.get(name=video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget = 0),
                system_instruction=sys_prompt
            ),
            contents=[
                types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
            ]
        )
        # Cleanup
        client.files.delete(name=video_file.name)

    try:
        result = json.loads(response.text)
    except Exception as e:
        print(f"[Error] Failed to parse JSON response: {e}")
        result = {"error": "Invalid JSON response from LLM", "raw_text": response.text}

    print(f'[Video Eval] Finished in {time.time() - start_t:.2f}s')
    return result



# def evaluate_video(video_path):
#     """Uploads video and performs visual analysis."""
#     print(f'[Video Eval] Starting video analysis for {video_path}...')
#     start_t = time.time()
    
#     prompt_path = os.path.join(PROMPT_DIR, "PitchAI_prompt_visual.yaml")
#     with open(prompt_path, "r") as f:
#         config = yaml.safe_load(f)

#     # 1. Upload
#     print(f'[Video Uploading] Uploading...')
#     video_file = client.files.upload(file=video_path)
    
#     # 2. Wait for Processing
#     while video_file.state.name == "PROCESSING":
#         time.sleep(1) # Reduced sleep time for faster polling
#         video_file = client.files.get(name=video_file.name)
    
#     if video_file.state.name == "FAILED":
#         raise ValueError("Video processing failed.")

#     print(f'[Video Eval] Processing complete. analyzing...')

#     # 3. Analyze
#     sys_prompt = config["system_prompt"].format(
#         TARGET_PERSONA="Not Provided",
#         DEAL_STAGE="Not Provided"
#     )

#     response = client.models.generate_content(
#         model="gemini-2.5-flash-lite",
#         config=types.GenerateContentConfig(
#             temperature=0.1,
#             response_mime_type="application/json",
#             thinking_config=types.ThinkingConfig(thinking_budget = 0)
#         ),
#         contents=[
#             types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
#             sys_prompt
#         ]
#     )
    
#     # 4. Cleanup
#     client.files.delete(name=video_file.name)
    
#     result = json.loads(response.text)
#     print(f'[Video Eval] Finished in {time.time() - start_t:.2f}s')
#     return result

def text_pipeline(audio_path):
    """Orchestrates the Audio -> Text -> Evaluation flow."""
    # transcript = llm_transcribe_audio(audio_path)
    transcript = transcribe_audio(audio_path)
    return evaluate_text(transcript)


def main():
    global_start = time.time()

    # Determine input type
    is_video = is_video_file(INPUT_PATH)
    print(f"--- Detected Input Type: {'VIDEO' if is_video else 'AUDIO'} ---")

    text_result = None
    video_result = None
    if is_video:
        # --- VIDEO FLOW ---
        # 1. Split
        audio_path, video_path = split_audio_video(INPUT_PATH)

        # 2. Parallel Execution
        print(f'\n--- Starting Parallel Execution ---')
        with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
            # Submit tasks
            future_text = executor.submit(text_pipeline, audio_path)
            future_video = executor.submit(evaluate_video, video_path)
            
            # Wait for results
            text_result = future_text.result()
            video_result = future_video.result()

    else:
        # --- AUDIO FLOW ---
        # 1. No Split required, pass input directly as audio path
        print(f'\n--- Starting Serial Execution (Audio Only) ---')
        # Note: faster_whisper handles mp3, wav, m4a etc directly
        text_result = text_pipeline(INPUT_PATH)

    # 3. Save Results
    print(f'\n--- Merging Results ---')
    with open("transcript_evaluation.json", "w") as f:
        json.dump(text_result, f, indent=4)

    if video_result:
        with open("video_evaluation.json", "w") as f:
            json.dump(video_result, f, indent=4)

    print(f"Total Workflow Time: {time.time() - global_start:.2f} seconds")

if __name__ == "__main__":
    main()