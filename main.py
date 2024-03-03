import os
import uuid
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import subprocess
import whisper
import soundfile as sf
import speechbrain
from pyannote.audio import Pipeline
"""
REQUIRED:

Populate access_token with a hugging face read token, instructions here:

# 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
# 2. visit hf.co/pyannote/segmentation and accept user conditions
# 3. visit hf.co/settings/tokens to create an access token
# 4. instantiate pretrained speaker diarization pipeline

"""
access_token = "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"

def ensure_dir(directory):
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_mp4_to_mp3(video_path, audio_path):
    """Converts MP4 video to MP3 audio."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='mp3')

def isolate_vocals_with_spleeter(audio_path, output_dir):
    """Uses Spleeter to isolate vocals from the audio."""
    ensure_dir(output_dir)
    subprocess.run(['python', '-m', 'spleeter', 'separate', '-p', 'spleeter:2stems', audio_path, '-o', output_dir], check=True)

def change_speed(audio_segment, speed=0.25):
    """Changes the playback speed of an audio segment."""
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * speed)
    }).set_frame_rate(audio_segment.frame_rate)

def split_audio(audio_path):
    """Splits audio into segments based on silence."""
    audio = AudioSegment.from_file(audio_path)
    slowed_audio = change_speed(audio)
    chunks = silence.split_on_silence(slowed_audio, min_silence_len=1000, silence_thresh=-40)
    return chunks

def save_segments(segments, output_dir):
    """Saves audio segments into a GUID-named folder."""
    folder_name = str(uuid.uuid4())
    folder_path = os.path.join(output_dir, folder_name)
    ensure_dir(folder_path)
    
    for i, segment in enumerate(segments):
        normal_speed_segment = change_speed(segment, speed=4.0)
        normal_speed_segment.export(os.path.join(folder_path, f"segment_{i+1}.mp3"), format="mp3")

def transcribe_and_filter_segments(segments_dir):
    """Transcribes audio segments using Whisper and deletes segments with fewer than two words."""
    model = whisper.load_model("medium")
    valid_segments = []
    
    for filename in os.listdir(segments_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(segments_dir, filename)
            result = model.transcribe(file_path)
            text = result["text"]
            
            if len(text.split()) < 2:
                os.remove(file_path)
                print(f"Deleted {filename} as it contains fewer than two words.")
            else:
                valid_segments.append(file_path)
    return valid_segments

def diarize_audio(audio_path, output_dir):
    """Performs speaker diarization on audio segments and saves them into speaker-specific subfolders."""
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=access_token)
    waveform, sample_rate = sf.read(audio_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_dir = os.path.join(output_dir, f"speaker_{speaker}")
        ensure_dir(speaker_dir)
        
        start_time, end_time = turn.start, turn.end
        start_sample, end_sample = int(start_time * sample_rate), int(end_time * sample_rate)
        speaker_waveform = waveform[start_sample:end_sample]
        
        speaker_segment_path = os.path.join(speaker_dir, f"{uuid.uuid4()}.wav")
        sf.write(speaker_segment_path, speaker_waveform, sample_rate)
        
    print(f"Diarization completed for {audio_path}. Results saved in {output_dir}.")

# Define paths
base_dir = "C:\\Users\\Jojo\\Desktop\\auto voice"
video_path = os.path.join(base_dir, "Unfiltered", "unfiltered.mp4")
temp_audio_path = os.path.join(base_dir, "temp_audio.mp3")  # Temporary path for the extracted audio
spleeter_output_dir = os.path.join(base_dir, "spleeter_output")
final_output_dir = os.path.join(base_dir, "filtered")

# Main process
convert_mp4_to_mp3(video_path, temp_audio_path)
isolate_vocals_with_spleeter(temp_audio_path, spleeter_output_dir)
vocals_path = os.path.join(spleeter_output_dir, os.path.splitext(os.path.basename(temp_audio_path))[0], "vocals.wav")
vocal_segments = split_audio(vocals_path)
save_segments(vocal_segments, final_output_dir)

# Transcribe, filter, and diarize
for guid_folder_name in os.listdir(final_output_dir):
    guid_folder_path = os.path.join(final_output_dir, guid_folder_name)
    valid_segments = transcribe_and_filter_segments(guid_folder_path)
    for audio_path in valid_segments:
        diarize_audio(audio_path, guid_folder_path)