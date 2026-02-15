from faster_whisper import WhisperModel
import time, os, json


model_size = "small.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(audio_path, output_json_path="transcript.json"):
    print("Starting transcription...")
    start_time = time.time()

    segments, info = model.transcribe(
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
        "full_text": ""
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

    transcript_data["full_text"] = " ".join(full_text_parts)

    # Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds.")
    print(f"Saved transcript to {output_json_path}")

    return transcript_data

# Usage
transcript = transcribe_audio(r"sample_pitch_audio_output.mp3")
print(transcript)