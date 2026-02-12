from faster_whisper import WhisperModel
import time

model_size = "large-v3"# small.en

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("sample_pitch_audio_output.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))




# SETTINGS
# 'small.en' is English-only and more accurate than generic 'small'
model_size = "small.en" 

# Run on CPU with INT8 quantization (Crucial for speed!)
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    print("Starting transcription...")
    start_time = time.time()

    # segments contains text + timestamps
    # info contains language detection (which we know is English)
    segments, info = model.transcribe(audio_path, beam_size=5)

    full_text = ""

    # Iterate through segments (this is where the processing happens)
    for segment in segments:
        # You can also capture timestamps here if you want to pass them to LLM
        # e.g., f"[{segment.start:.2f}] {segment.text}"
        full_text += segment.text + " "

    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds.")
    
    return full_text.strip()

# Usage
transcript = transcribe_audio("sample_pitch_audio_output.mp3")
print(transcript)