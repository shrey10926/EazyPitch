import os, json, time, yaml, subprocess, mimetypes, concurrent.futures, pandas as pd, numpy as np, re, sys
import gradio as gr
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

# Assuming these are your custom modules in the same directory
from redact import Redactor
from db_ops import get_all_data_as_dataframe


if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport

    # Save the original method
    _original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
    
    # Define a patched version that ignores WinError 10054
    def _patched_call_connection_lost(self, exc):
        try:
            _original_call_connection_lost(self, exc)
        except ConnectionResetError as e:
            if getattr(e, 'winerror', None) == 10054:
                pass  # Ignore this specific harmless error
            else:
                raise
                
    # Apply the patch
    _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
# ==========================================


# --- LOAD CONFIGURATION ---
try:
    with open(r"app_config.yaml", "r") as f:
        APP_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("app_config.yaml not found! Please ensure it is in the same directory.")

# --- API CLIENT SETUP ---
# It's best practice to use environment variables, but you can hardcode it here for the demo.
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
client = genai.Client(api_key=API_KEY, http_options={'timeout': APP_CONFIG['api']['timeout_seconds'] * 1000})

# --- STATIC SETUP ---
print("Loading Whisper Model... This might take a moment.")
transcribe_model = WhisperModel(
    APP_CONFIG['whisper']['model_size'], 
    device=APP_CONFIG['whisper']['device'], 
    compute_type=APP_CONFIG['whisper']['compute_type']
)
redactor = Redactor()


# ==========================================
# HELPER FUNCTIONS (From your original code)
# ==========================================
def ensure_path(d, *keys):
    cur = d
    for k in keys:
        cur = cur.setdefault(k, {})
    return cur

def ensure_defaults(df, columns, default='Not Provided'):
    for col in columns:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            df[col] = df[col].fillna(default).astype(str)
    return df

def is_video_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('video'):
        return True
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def split_audio_video(input_path, output_audio_path, output_video_path):
    print(f'[FFMPEG] Splitting audio and video...')
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, 
            "-vn", "-acodec", "libmp3lame", output_audio_path,
            "-an", "-vf", "scale=640:360,fps=5", "-c:v", "libx264", "-preset", "ultrafast", output_video_path
        ], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFMPEG processing failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFMPEG not found in system path.")

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def transcribe_audio(audio_path, transcript_path):
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

    transcript_data["full_transcript"] = redactor.redact_pii(" ".join(full_text_parts))

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    return {key: transcript_data[key] for key in ["full_transcript"]}

def evaluate_text(client, transcript_data, prompt_dir, schema_path, company_ID, context_df):
    prompt_path = os.path.join(prompt_dir, "Text_Prompt_2.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)

    with open(schema_path, 'r') as file:
        text_response_schema = json.load(file)

    context_vars = {k: "Not Provided" for k in['product_name', 'target_persona', 'deal_stage', 'pain_points', 'USP', 'competitors', 'mandatory_terms', 'forbidden_terms']}
    row = context_df[context_df['company_ID'] == company_ID]

    if not row.empty:
        for key in context_vars.keys():
            context_vars[key] = row.iloc[0].get(key.lower(), 'Not Provided')

    context_vars["transcript"] = transcript_data["full_transcript"]

    text_sys_prompt = config["system_prompt"]
    text_user_prompt = config["user_prompt"].format(**context_vars)

    api_config = types.GenerateContentConfig(
        temperature=APP_CONFIG['api']['temperature'],
        response_mime_type="application/json",
        response_schema=text_response_schema,
        system_instruction=text_sys_prompt
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
                time.sleep(APP_CONFIG['api']['retry_delay'])
            else:
                return {"text_error": f"API failed: {str(e)}"}

    try:
        raw_text = response.text.strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        result = json.loads(raw_text.strip())
    except Exception as e:
        return {"text_error": "Invalid JSON response from LLM", "raw_text": getattr(response, 'text', 'No response text')}

    return result

def evaluate_video(client, video_path, prompt_dir, schema_path):
    prompt_path = os.path.join(prompt_dir, "PitchAI_prompt_visual.yaml")
    with open(prompt_path, "r") as f:
        sys_prompt = yaml.safe_load(f)["system_prompt"]

    with open(schema_path, 'r') as file:
        video_response_schema = json.load(file)

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    api_config = types.GenerateContentConfig(
        temperature=APP_CONFIG['api']['temperature'],
        response_mime_type="application/json",
        response_schema=video_response_schema,
        system_instruction=sys_prompt
    )

    if file_size_mb < APP_CONFIG['video']['max_inline_size_mb']:
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
            return {"video_error": f"API generation failed: {str(e)}"}
    else:
        video_file = client.files.upload(file=video_path)
        try:
            timeout_seconds = APP_CONFIG['api']['timeout_seconds']
            start_poll_time = time.time()

            while (video_file.state.name if hasattr(video_file.state, 'name') else video_file.state) == "PROCESSING":
                if time.time() - start_poll_time > timeout_seconds:
                    raise TimeoutError(f"Video Processing timed out!")
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)
            
            response = client.models.generate_content(
                model=APP_CONFIG['api']['model_name'],
                config=api_config,
                contents=[types.Part.from_text(text="Analyze this video's visual delivery based on the system instructions."),
                          types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)]
            )
        except Exception as e:
            return {"video_error": f"API generation failed: {str(e)}"}
        finally: 
            client.files.delete(name=video_file.name)

    try:
        raw_text = response.text.strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return json.loads(raw_text.strip())
    except Exception as e:
        return {"video_error": "Invalid JSON response", "raw_text": getattr(response, 'text', 'No response text')}

def text_pipeline(client, audio_path, transcript_path, prompt_dir, text_schema_path, company_ID, context_df):
    transcript = transcribe_audio(audio_path, transcript_path)
    eval_result = evaluate_text(client, transcript, prompt_dir, text_schema_path, company_ID, context_df)
    return transcript, eval_result

def video_pipeline(client, output_video_path, prompt_dir, video_schema_path):
    return evaluate_video(client, output_video_path, prompt_dir, video_schema_path)



def process_pitch_gradio(video_file, audio_file, progress=gr.Progress()):
    """Main entry point for the Gradio UI."""
    
    # Determine input logic
    input_path = video_file if video_file else audio_file
    if not input_path:
        return "No file provided.", {"error": "Upload a file"}, {"error": "Upload a file"}
    
    # Extract filename and company_ID
    filename_with_ext = os.path.basename(input_path)
    company_ID, _ = os.path.splitext(filename_with_ext)

    progress(0.1, desc=f"Setting up directories for '{filename_with_ext}'...")

    # --- CHANGED: Setup Output Directory based on the uploaded file's name ---
    output_dir = os.path.join("Output", company_ID)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all output file paths inside the new directory
    transcript_path = os.path.join(output_dir, "transcript.json")
    text_eval_path = os.path.join(output_dir, "text_eval.json")
    video_eval_path = os.path.join(output_dir, "video_eval.json")

    # Hardcoded paths from your script
    prompt_dir = r"Prompts"
    text_schema_path = os.path.join("Response_Schema", "text_response_schema_2.json")
    video_schema_path = os.path.join("Response_Schema", "video_response_schema.json")

    # Fetch Data Context
    df = get_all_data_as_dataframe()
    df['company_ID'] = df['company_ID'].astype('str')
    df = ensure_defaults(df,[
        'product_name','target_persona','deal_stage','pain_points',
        'USP','competitors','mandatory_terms','forbidden_terms'
    ])

    is_video = is_video_file(input_path)
    text_result, video_result, transcript_text = {}, {}, "No transcript generated."

    if is_video:
        progress(0.2, desc="Agent is analyzing the pitch....")
        video_file_path = os.path.join(output_dir, "video_output.mp4")
        audio_file_path = os.path.join(output_dir, "audio_output.mp3")
        split_audio_video(input_path, audio_file_path, video_file_path)

        progress(0.4, desc="Agent is analyzing the pitch....")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(text_pipeline, client, audio_file_path, transcript_path, prompt_dir, text_schema_path, company_ID, df)
            future_video = executor.submit(video_pipeline, client, video_file_path, prompt_dir, video_schema_path)

            timeout = APP_CONFIG['api']['timeout_seconds'] + 60
            try:
                transcript_data, text_result = future_text.result(timeout=timeout)
                transcript_text = transcript_data.get("full_transcript", "")
            except Exception as e:
                text_result = {"error": f"Critical Text pipeline failure: {e}"}

            try:
                video_result = future_video.result(timeout=timeout)
            except Exception as e:
                video_result = {"error": f"Critical Video pipeline failure: {e}"}

    else:
        progress(0.4, desc="Agent is analyzing the pitch....")
        try:
            transcript_data, text_result = text_pipeline(client, input_path, transcript_path, prompt_dir, text_schema_path, company_ID, df)
            transcript_text = transcript_data.get("full_transcript", "")
            video_result = {"info": "Audio-only input detected. Visual evaluation skipped."}
        except Exception as e:
            text_result = {"error": str(e)}

    # --- NEW: Save Evaluation Results to JSON files in the Output folder ---
    progress(0.9, desc="Saving Evaluation files...")

    with open(text_eval_path, "w", encoding="utf-8") as f:
        json.dump(text_result, f, indent=4, ensure_ascii=False)
        
    with open(video_eval_path, "w", encoding="utf-8") as f:
        json.dump(video_result, f, indent=4, ensure_ascii=False)

    progress(1.0, desc="Done!")
    return transcript_text, text_result, video_result


# ==========================================
# GRADIO UI DEFINITION
# ==========================================

# Define a clean, modern, minimalistic theme
modern_theme = gr.themes.Default(
    primary_hue="slate",     # Professional, muted primary color
    neutral_hue="zinc",      # Clean grays for backgrounds
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
    block_label_text_weight="600"
)


custom_css = """
.center-header { text-align: center; margin-bottom: 20px; }
.center-header h1 { font-weight: 700; margin-bottom: 5px; font-size: 2.2em;}
.center-header p { color: #666; font-size: 1.1em; }
.helper-text { text-align: center; font-size: 0.85em; color: gray; margin-top: 10px; }
"""

with gr.Blocks(title="Sales Pitch AI Evaluator", theme=modern_theme, css=custom_css) as demo:
    
    # Centered Minimalistic Header
    gr.HTML(
        """
        <div class="center-header">
            <h1>🎙️ Sales Pitch AI Evaluator</h1>
            <p>Upload or record a pitch for instant AI feedback on strategy, structure, and delivery.</p>
        </div>
        """
    )

    with gr.Row():
        
        # LEFT COLUMN: Inputs (Wrapped in a panel for a clean card look)
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Input Media")
            
            with gr.Tabs():
                with gr.Tab("📹 Video"):
                    video_input = gr.Video(
                        label="Upload or Record Video", 
                        sources=["upload", "webcam"],
                        show_label=False # Hides redundant label for minimalism
                    )
                with gr.Tab("🎧 Audio"):
                    audio_input = gr.Audio(
                        label="Upload or Record Audio", 
                        sources=["upload", "microphone"],
                        type="filepath",
                        show_label=False
                    )
            
            analyze_btn = gr.Button("Analyze Pitch", variant="primary", size="lg")
            gr.HTML("<div class='helper-text'>Large video processing might take 1-2 minutes depending on file size.</div>")

        # RIGHT COLUMN: Outputs (Symmetrical 50/50 split, using Tabs for cleanliness)
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 2. Evaluation Results")
            
            with gr.Tabs():
                with gr.Tab("🧠 Strategy & Structure"):
                    text_eval_output = gr.JSON(
                        label = "Strategic Analysis",
                        show_label=False

                    )
                
                with gr.Tab("🎥 Visual Delivery"):
                    video_eval_output = gr.JSON(
                        label="Visual Analysis",
                        show_label=False
                    )
                    
                with gr.Tab("📝 Redacted Transcript"):
                    transcript_output = gr.Textbox(
                        label="Transcribed Text (PII Redacted)", 
                        lines=20, 
                        interactive=False,
                        show_label=False,
                        placeholder="Transcript will appear here after processing..."
                    )

    # Button Logic
    analyze_btn.click(
        fn=process_pitch_gradio,
        inputs=[video_input, audio_input],
        outputs=[transcript_output, text_eval_output, video_eval_output]
    )

# --- RUN APP ---
if __name__ == "__main__":
    demo.launch(
        share=False, 
        server_name="0.0.0.0",
        server_port=6996
    )