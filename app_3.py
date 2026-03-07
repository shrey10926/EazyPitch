import os, json, time, yaml, subprocess, mimetypes, concurrent.futures, pandas as pd, numpy as np
from faster_whisper import WhisperModel
from google import genai
from google.genai import types


# --- LOAD CONFIGURATION ---
with open(r"app_config.yaml", "r") as f:
    APP_CONFIG = yaml.safe_load(f)


# --- CONFIGURATION (Keep only static setups here) ---
API_KEY = ""  # Replace with actual key
model_size = "small.en"
transcribe_model = WhisperModel(
    APP_CONFIG['whisper']['model_size'], 
    device=APP_CONFIG['whisper']['device'], 
    compute_type=APP_CONFIG['whisper']['compute_type']
)
client = genai.Client(api_key=API_KEY)


def ensure_path(d, *keys):
    cur = d
    for k in keys:
        cur = cur.setdefault(k, {})
    return cur


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


def split_audio_video(input_path, output_video_path):
    """SplStrips audio from video file."""
    print(f'[FFMPEG] Stripping audio from video...')

    subprocess.run([
            "ffmpeg", "-y", "-i", input_path, 
            "-an", "-vf", "scale=640:360,fps=5", "-c:v", "libx264", "-preset", "ultrafast", output_video_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    # subprocess.run([
    #     "ffmpeg", "-y", "-i", input_path, 
    #     "-vn", "-acodec", "libmp3lame", AUDIO_FILE_PATH,  # Output 1: Audio
    #     "-an", "-vf", "scale=640:360,fps=5", "-c:v", "libx264", "-preset", "ultrafast", VIDEO_FILE_PATH # Output 2: Video
    # ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f'[FFMPEG] Split complete.')
    return f"Successfully stripped the audio from video!"


def ensure_defaults(df, columns, default='Not Provided'):
    for col in columns:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            df[col] = df[col].fillna(default).astype(str)
    return df




def llm_transcribe_audio(audio_path, prompt_dir):
    """Sends audio to Gemini for transcription."""
    print(f'[Transcribe] Starting transcription for {audio_path}...')
    start_t = time.time()

    # Load Prompt
    prompt_path = os.path.join(prompt_dir, "PitchAI_prompt_transcribe.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)
    
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=config["system_prompt"]
        ),
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
        ]
    )

    transcript_data = json.loads(response.text)
    
    # Save intermediate result
    base_name = os.path.splitext(audio_path)[0].replace("_audio_output", "")
    with open(f"{base_name}_transcript.json", 'w') as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)
        
    print(f'[Transcribe] Finished in {time.time() - start_t:.2f}s')
    return transcript_data


def transcribe_audio(audio_path, transcript_path):
    print("Starting transcription...")
    start_time = time.time()

    segments, info = transcribe_model.transcribe(
        audio_path,
        beam_size=APP_CONFIG['whisper']['beam_size'],
        word_timestamps=False,
        language=APP_CONFIG['whisper']['language'],
        task=APP_CONFIG['whisper']['task'],
        temperature=APP_CONFIG['whisper']['temperature'],
        vad_filter=True
    )

    transcript_data = {
        "language": info.language,
        "duration": info.duration,
        "segments":[],
        "full_transcript": ""
    }

    full_text_parts =[]
    for segment in segments:
        segment_dict = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }

        transcript_data["segments"].append(segment_dict)
        full_text_parts.append(segment.text.strip())

    transcript_data["full_transcript"] = " ".join(full_text_parts)

    # Save JSON using the passed argument
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    full_transcript = {key: transcript_data[key] for key in ["full_transcript"]}

    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds.")
    print(f"Saved transcript to {transcript_path}")

    return full_transcript


def evaluate_text(transcript_data, prompt_dir, schema_path, company_ID, context_df):
    """Analyzes the transcript JSON."""
    print(f'[Text Eval] Starting text analysis...')
    start_t = time.time()

    prompt_path = os.path.join(prompt_dir, "PitchAI_prompt_text.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)


    try:
        # Cross-platform safe path opening
        with open(schema_path, 'r') as file:
            text_response_schema = json.load(file)
    except FileNotFoundError:
        print("Error: The text response schema file was not found. Please check the file path.")
        return {"text_error": "Schema file missing. Halting text evaluation."}
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. The file may be malformed.")
        return {"text_error": "Schema JSON malformed. Halting text evaluation."}

    context_vars = {k: "Not Provided" for k in['PRODUCT_NAME', 'TARGET_PERSONA', 'DEAL_STAGE', 'PAIN_POINTS', 'USP', 'COMPETITORS', 'MANDATORY_TERMS', 'FORBIDDEN_TERMS']}

    row = context_df[context_df['company_ID'] == company_ID]

    if not row.empty:
        for key in context_vars.keys():
            context_vars[key] = row.iloc[0].get(key.lower(), 'Not Provided')

    # if not row.empty:
    #     row_data = row.iloc[0]

    #     context_vars['PRODUCT_NAME'] = row_data.get('product_name', 'Not Provided')
    #     context_vars['TARGET_PERSONA'] = row_data.get('target_persona', 'Not Provided')
    #     context_vars['DEAL_STAGE'] = row_data.get('deal_stage', 'Not Provided')
    #     context_vars['PAIN_POINTS'] = row_data.get('pain_points', 'Not Provided')
    #     context_vars['USP'] = row_data.get('usp', 'Not Provided')
    #     context_vars['COMPETITORS'] = row_data.get('competitors', 'Not Provided')
    #     context_vars['MANDATORY_TERMS'] = row_data.get('mandatory_terms', 'Not Provided')
    #     context_vars['FORBIDDEN_TERMS'] = row_data.get('forbidden_terms', 'Not Provided')
    # else:
    #     print(f'[Warning] Company ID {company_ID} not found in database. Using defaults!')

    # Inject the dynamic context in system prompt
    text_sys_prompt = config["system_prompt"].format(**context_vars)

    # Prepare inputs
    transcript_str = json.dumps(transcript_data, indent=2, ensure_ascii=False)
    text_user_prompt = config["user_prompt"].format(transcript=transcript_str)

    # Fix #9: Centralized config and HTTP timeout inclusion
    api_config = types.GenerateContentConfig(
        temperature=APP_CONFIG['api']['temperature'],
        response_mime_type="application/json",
        response_schema=text_response_schema,
        system_instruction=text_sys_prompt,
        # http_options={'timeout': APP_CONFIG['api']['timeout_seconds']}
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=APP_CONFIG['api']['model_name'],
                config=api_config,
                contents=[types.Part.from_text(text=text_user_prompt)]
            )
            break
        except Exception as e:
            if attempt == 0:
                print(f"[text_Error] Generation failed: {e}. Retrying...")
                time.sleep(APP_CONFIG['api']['retry_delay'])
            else:
                return {"text_error": f"API failed: {str(e)}"}

    # result = parse_llm_response(response, text_response_schema)

    try:
        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:] 
        if raw_text.startswith("```"):
            raw_text = raw_text[3:] 
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3] 

        result = json.loads(raw_text.strip())

    except Exception as e:
        print(f"[text_Error] Failed to parse JSON response: {e}")
        result = {"text_error": "Invalid JSON response from LLM", "raw_text": response.text}

    if 'text_error' not in result:
        strategic = result.get('strategic_analysis') or {}
        structure = result.get('structure_analysis') or {}

        persona_fit_score = strategic.get('persona_fit_score')
        structure_score = structure.get('score')

        weights, scores = [], []
        if isinstance(persona_fit_score, (int, float)):
            weights.append(0.6)
            scores.append(persona_fit_score)

        if isinstance(structure_score, (int, float)):
            weights.append(0.4)
            scores.append(structure_score)

        if scores:
            normalized_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
            overall_score = round(normalized_score, 2)
        else:
            overall_score = None  # Everything was skipped

        ensure_path(result, 'text_summary')
        result['text_summary']['overall_score'] = overall_score

        return result
    else:
        return result



def evaluate_video(video_path, prompt_dir, schema_path):
    """
    Hybrid Logic:
    - If video < 14MB: Send Inline (Fastest, ~30s latency)
    - If video >= 14MB: Upload via File API (Slower, but won't crash)
    """
    print(f'[Video Eval] Starting video analysis for {video_path}...')
    start_t = time.time()
    
    prompt_path = os.path.join(prompt_dir, "PitchAI_prompt_visual.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)

    sys_prompt = config["system_prompt"]

    try:
        # Cross-platform safe path opening
        with open(schema_path, 'r') as file:
            video_response_schema = json.load(file)
    except FileNotFoundError:
        print("Error: The video response schema file was not found. Please check the file path.")
        return {"video_error": "video response schema file missing. Halting video evaluation."}
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. The file may be malformed.")
        return {"video_error": "Schema JSON malformed. Halting video evaluation."}


    # 1. Check File Size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f'[Video Eval] Optimized Video Size: {file_size_mb:.2f} MB')

    api_config = types.GenerateContentConfig(
        temperature=APP_CONFIG['api']['temperature'],
        response_mime_type="application/json",
        response_schema=video_response_schema,
        system_instruction=sys_prompt,
        # http_options={'timeout': APP_CONFIG['api']['timeout_seconds']}
    )

    # THRESHOLD: 14MB (Leaving 2MB buffer for prompt text and overhead)
    if file_size_mb < APP_CONFIG['video']['max_inline_size_mb']:
        # --- METHOD A: INLINE (FAST) ---
        print(f'[Video Eval] Size is under limit. Using INLINE mode (Fast)...')
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        try:
            response = client.models.generate_content(
                    model=APP_CONFIG['api']['model_name'],
                    config=api_config,
                    contents=[
                        types.Part.from_text(text="Analyze visual delivery based on system instructions."),
                        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
                    ]
                        )
        except Exception as e:
            print(f"[video_Error] Gemini API generation failed: {e}")
            return {"video_error": f"API generation failed: {str(e)}"}

    else:
        # --- METHOD B: FILE API (SAFE) ---
        print(f'[Video Eval] Size is over limit. Using FILE API mode (Safe)...')
        video_file = client.files.upload(file=video_path)

        try:
            timeout_seconds = APP_CONFIG['api']['timeout_seconds']
            start_poll_time = time.time()

            # Helper to safely get the state
            def get_state(v_file):
                return v_file.state.name if hasattr(v_file.state, 'name') else v_file.state

            # Fast Polling
            while get_state(video_file) == "PROCESSING":
                if time.time() - start_poll_time > timeout_seconds:
                    raise TimeoutError(f"Video Processing timed out on Google's server!")
                
                print(f" Video is processing, waiting...")
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed.")
            
            response = client.models.generate_content(
                model=APP_CONFIG['api']['model_name'],
                config=api_config,
                contents=[types.Part.from_text(text="Analyze this video's visual delivery based on the system instructions."),
                    types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
                ]
            )

        except Exception as e:
            print(f"[video_Error] API generation failed: {e}")
            return {"video_error": f"API generation failed: {str(e)}"}

        finally: 
            print(f"[Video Eval] Cleaning up API file: {video_file.name}")
            client.files.delete(name=video_file.name)

    try:
        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:] 
        if raw_text.startswith("```"):
            raw_text = raw_text[3:] 
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3] 

        result = json.loads(raw_text.strip())

    except Exception as e:
        print(f"[video_Error] Failed to parse JSON response: {e}")
        result = {"video_error": "Invalid JSON response from LLM", "raw_text": getattr(response, 'text', 'No response text')}

    print(f'[Video Eval] Finished in {time.time() - start_t:.2f}s')
    return result


def text_pipeline(audio_path, transcript_path, prompt_dir, text_schema_path, company_ID, context_df):
    """Orchestrates the Audio -> Text -> Evaluation flow."""
    transcript = transcribe_audio(audio_path, transcript_path)
    return evaluate_text(transcript, prompt_dir, text_schema_path, company_ID, context_df)


def video_pipeline(input_video_path, output_video_path, prompt_dir, video_schema_path):
    """Orchestrates the Video Split -> Evaluation flow."""
    split_audio_video(input_video_path, output_video_path)
    return evaluate_video(output_video_path, prompt_dir, video_schema_path)


def main():
    global_start = time.time()

    # --- DYNAMIC CONFIGURATION & PATH GENERATION ---
    input_path = r"sample_pitch.mp4"
    file_name = os.path.basename(input_path)
    output_dir = os.path.join("Output", os.path.splitext(file_name)[0])
    os.makedirs(output_dir, exist_ok=True)

    text_result_path = os.path.join(output_dir, "text_evaluation.json")
    transcript_path = os.path.join(output_dir, "transcript.json")
    video_result_path = os.path.join(output_dir, "video_evaluation.json")
    video_file_path = os.path.join(output_dir, "video_output.mp4")

    prompt_dir = r"Prompts"
    text_schema_path = os.path.join("Response_Schema", "text_response_schema.json")
    video_schema_path = os.path.join("Response_Schema", "video_response_schema.json")
    # ------------------------------------------------

    df = pd.read_excel(r'Context.xlsx')
    df['company_ID'] = df['company_ID'].astype('str')
    company_ID = os.path.splitext(file_name)[0]
    df = ensure_defaults(df, [
        'product_name','target_persona','deal_stage','pain_points',
        'usp','competitors','mandatory_terms','forbidden_terms'
    ])

    # Determine input type
    is_video = is_video_file(input_path)
    print(f"--- Detected Input Type: {'VIDEO' if is_video else 'AUDIO'} ---")

    text_result, video_result = None, None
    if is_video:
        # --- VIDEO FLOW ---
        print(f'\n--- Starting Parallel Execution ---')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Pass the specific variables each pipeline needs
            # audio_path, transcript_path, prompt_dir, text_schema_path, company_ID, context_df
            future_text = executor.submit(text_pipeline, input_path, transcript_path, prompt_dir, text_schema_path, company_ID, df)
            future_video = executor.submit(video_pipeline, input_path, video_file_path, prompt_dir, video_schema_path)

            # Wait for results
            # text_result = future_text.result()
            # video_result = future_video.result()

            # Fix #13: Global Error boundaries for Futures with Timeout (Fix #9)
            timeout = APP_CONFIG['api']['timeout_seconds'] + 60
            try:
                text_result = future_text.result(timeout=timeout)
            except Exception as e:
                print(f"[CRITICAL] Text pipeline crashed: {e}")
                text_result = {"error": f"Critical pipeline failure: {e}"}

            try:
                video_result = future_video.result(timeout=timeout)
            except Exception as e:
                print(f"[CRITICAL] Video pipeline crashed: {e}")
                video_result = {"error": f"Critical pipeline failure: {e}"}

    else:
        try:
            # --- AUDIO FLOW ---
            print(f'\n--- Starting Serial Execution (Audio Only) ---')
            # Pass the specific variables the text pipeline needs
            text_result = text_pipeline(input_path, transcript_path, prompt_dir, text_schema_path, company_ID, df)
        except Exception as e:
            print(f"[CRITICAL] Audio pipeline crashed: {e}")
            text_result = {"error": str(e)}

    # Save Results
    print(f'\n--- Merging Results ---')
    with open(text_result_path, "w") as f:
        json.dump(text_result, f, indent=4, ensure_ascii=False)

    if video_result:
        with open(video_result_path, "w") as f:
            json.dump(video_result, f, indent=4, ensure_ascii=False)

    print(f"Total Workflow Time: {time.time() - global_start:.2f} seconds")


if __name__ == "__main__":
    main()