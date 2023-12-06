#pip install yt_dlp
#pip install pydub
import yt_dlp

URLS = 'https://www.youtube.com/watch?v=y9k-U9AuDeM'

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(URLS)
    info = ydl.extract_info(URLS)
    filename = ydl.prepare_filename(info)

import whisper
model = whisper.load_model("base")
result = model.transcribe(filename.replace('\\',''), fp16=False)

