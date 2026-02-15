import os, json, time, yaml, subprocess
from google import genai
from google.genai import types

client = genai.Client(api_key = "")
# pip install google-genai PyYAML pandas openpyxl numpy
# python.exe -m pip install --upgrade pip

start = time.time()

############################################################################################################################
# SPLIT AUDIO/VIDEO
############################################################################################################################
print(f'Splitting audio and video...')
input_video = r"sample_pitch.mp4"
output_audio = fr"{os.path.splitext(input_video)[0]}_audio_output.mp3"
output_video = fr"{os.path.splitext(input_video)[0]}_video_output.mp4"

audio_command = [
    "ffmpeg",
    "-y",                  # overwrite output
    "-i", input_video,
    output_audio
]

video_command = [
    "ffmpeg",
    "-y",                  # overwrite output
    "-i", input_video,
    "-an", 
    "-c:v",
    "copy",
    output_video
]

subprocess.run(audio_command, check=True)
subprocess.run(video_command, check=True)

print(f'Audio and video split successfully.')
print(f'time taken to split audio video : {time.time() - start} seconds')

############################################################################################################################
# TRANSCRIBE
############################################################################################################################
print(f'Transcribing audio...')


with open(r"Prompts\PitchAI_prompt_transcribe.yaml", "r") as f:
    transcribe_prompt = yaml.safe_load(f)

transcript_system_prompt = transcribe_prompt["system_prompt"]
# user_prompt = transcribe_prompt["user_prompt"]



with open(output_audio, 'rb') as f:
    audio_bytes = f.read()

transcript_response = client.models.generate_content(
  model = "gemini-2.5-flash-lite",
  config = types.GenerateContentConfig(
        temperature = 0.1,
        seed = 69,
        response_mime_type = "application/json",
        thinking_config = types.ThinkingConfig(thinking_budget = 0), # 0 disable thinking, -1 dynamic thinking
    ),
  contents = [
    types.Part.from_bytes(
      data = audio_bytes,
      mime_type = "audio/mp3",
    ), transcript_system_prompt
  ]
)

# Parse response to ensure it's valid JSON before saving
transcript_data = json.loads(transcript_response.text)

with open(fr"{os.path.splitext(input_video)[0]}_transcript.json", 'w') as json_file:
    json.dump(transcript_data, json_file, indent=4)

print(f'Transcription complete.')
print(f'time taken to transcribe audio : {time.time() - start} seconds')

############################################################################################################################
# TEXT EVALUATION
############################################################################################################################
print(f'Evaluating transcript...')

with open(r"Prompts\PitchAI_prompt_text.yaml", "r") as f:
    prompt = yaml.safe_load(f)

system_prompt = prompt["system_prompt"]
user_prompt = prompt["user_prompt"]

with open(fr"{os.path.splitext(input_video)[0]}_transcript.json", 'r') as file:
    transcript = json.load(file)

# Convert json to str to replace single to double quotes
transcript_str = json.dumps(transcript, indent=2)

transcript_evaluation = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    config = types.GenerateContentConfig(
        temperature = 0.1,
        seed = 69,
        response_mime_type = "application/json",
        thinking_config = types.ThinkingConfig(thinking_budget = -1),
        system_instruction = system_prompt.format(
        PRODUCT_NAME = "Not Provided",
        TARGET_PERSONA = "Not Provided",
        DEAL_STAGE = "Not Provided",
        PAIN_POINTS = "Not Provided",
        USP = "Not Provided",
        COMPETITORS = "Not Provided",
        MANDATORY_TERMS = "Not Provided",
        FORBIDDEN_TERMS = "Not Provided",
    )),
    contents=[user_prompt.format(transcript = transcript_str)],
    )

with open(r"transcript_evaluation.json", 'w') as json_file:
    json.dump(json.loads(transcript_evaluation.text), json_file, indent=4)

print(f'Transcript evaluation complete.')
print(f'time taken to evaluate text : {time.time() - start} seconds')
############################################################################################################################
# VIDEO EVALUATION
############################################################################################################################
print(f'Evaluating video...')

with open(r"Prompts\PitchAI_prompt_visual.yaml", "r") as f:
    visual_prompt = yaml.safe_load(f)

visual_system_prompt = visual_prompt["system_prompt"]
# visual_user_prompt = visual_prompt["user_prompt"]


print(f'uploading video file...')
video_file = client.files.upload(file = output_video)
# audio_file = client.files.upload(file=r"output.mp3")


# Wait for file to be active (Good practice for videos)
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

print(f'\nVideo file `{video_file.name}` is ready.')

print(f'Starting Video Evaluation...')
video_evaluation = client.models.generate_content(
    model="gemini-2.5-flash-lite", #gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3-flash-preview
    config = types.GenerateContentConfig(
        temperature = 0.1,
        seed = 69,
        response_mime_type = "application/json",
        thinking_config = types.ThinkingConfig(thinking_budget = -1), # 0 disable thinking, -1 dynamic thinking
    ),
    contents=[types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type), 
    visual_system_prompt.format(
        TARGET_PERSONA = "Not Provided",
        DEAL_STAGE = "Not Provided"
    )],
)

with open(r"video_evaluation.json", 'w') as json_file:
    json.dump(json.loads(video_evaluation.text), json_file, indent=4)

print(f'deleting video file...')
client.files.delete(name = video_file.name)
print(f'video file deleted successfully.')

print(f'Video evaluation complete.')
print(f'time taken to evaluate video : {time.time() - start} seconds')
end = time.time()

print(f"Total time taken: {end - start} seconds")