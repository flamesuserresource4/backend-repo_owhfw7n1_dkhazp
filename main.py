import os
import io
import json
import shutil
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Defer heavy imports to runtime where possible
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # will attempt to import at request time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
whisper_model = None


def safe_word(word: str) -> str:
    return word.replace('{', '｛').replace('}', '｝')


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ffprobe_size(video_path: str) -> tuple[int, int]:
    # Use ffprobe to read width/height
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    try:
        proc = run_cmd(cmd)
        data = json.loads(proc.stdout.decode("utf-8"))
        stream = data.get("streams", [{}])[0]
        return int(stream.get("width", 1080)), int(stream.get("height", 1920))
    except Exception:
        # Fallback to portrait reel size
        return 1080, 1920


def extract_audio(video_path: str, audio_path: str) -> None:
    # Extract audio track to mp3; if no audio, this will fail
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-b:a", "128k",
        audio_path,
    ]
    run_cmd(cmd)


def build_ass_style_line(
    *,
    width: int,
    height: int,
    fontname: str = "Arial",
    fontsize: int = 45,
    primary_colour: str = "&H00FFFF00",
    secondary_colour: str = "&H00FFFF00",
    outline_colour: str = "&H00000000",
    back_colour: str = "&H00000000",
    outline_width: int = 2,
    shadow_depth: int = 3,
    margin_percent: int = 15,
    vertical_percent: int = 12,
):
    margin_l = margin_r = int(width * margin_percent / 100)
    margin_v = int(height * vertical_percent / 100)

    style_line = (
        f"Style: Default,{fontname},{fontsize},{primary_colour},{secondary_colour},{outline_colour},{back_colour},"
        f"1,0,0,0,100,100,0,0,1,{outline_width},{shadow_depth},2,{margin_l},{margin_r},{margin_v},1"
    )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


def fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t - int(t)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def subtitle_video_oneword(
    video_path: str,
    *,
    model,
    fontname: str = "Arial",
    fontsize: int = 45,
    primary_colour: str = "&H00FFFF00",
    secondary_colour: str = "&H00FFFF00",
    outline_colour: str = "&H00000000",
    back_colour: str = "&H00000000",
    outline_width: int = 2,
    shadow_depth: int = 3,
    margin_percent: int = 15,
    vertical_percent: int = 12,
) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.mp3")
        ass_path = os.path.join(temp_dir, "subtitles.ass")
        output_path = os.path.join(temp_dir, "output.mp4")

        # Extract audio and get video size without moviepy
        width, height = ffprobe_size(video_path)
        try:
            extract_audio(video_path, audio_path)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail="Failed to extract audio (video may have no audio track)")

        if model is None:
            raise HTTPException(status_code=500, detail="Whisper model failed to load.")
        result = model.transcribe(audio_path, word_timestamps=True)

        ass_header = build_ass_style_line(
            width=width,
            height=height,
            fontname=fontname,
            fontsize=fontsize,
            primary_colour=primary_colour,
            secondary_colour=secondary_colour,
            outline_colour=outline_colour,
            back_colour=back_colour,
            outline_width=outline_width,
            shadow_depth=shadow_depth,
            margin_percent=margin_percent,
            vertical_percent=vertical_percent,
        )

        lines = []
        for seg in result.get('segments', []):
            words = seg.get("words") if isinstance(seg, dict) else None
            if not words:
                start = seg.get('start', 0)
                end = seg.get('end', start + 1)
                text = seg.get('text', '').strip().replace('\n', ' ')
                lines.append(
                    f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{{\\bord3\\shad1}}{text}"
                )
            else:
                for w in words:
                    start = float(w['start'])
                    end = max(float(w['end']), float(w['start']) + 0.04)
                    # Primary colour already in ASS format
                    text = f"{{\\bord3\\shad1}}{{\\c{primary_colour}&}}{safe_word(w['word'])}"
                    lines.append(
                        f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{text}"
                    )

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
            for line in lines:
                f.write(line + "\n")

        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"ass={ass_path}",
            "-c:a", "copy",
            "-y",
            output_path,
        ]
        try:
            run_cmd(command)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"FFmpeg failed: {e.stderr.decode('utf-8')[:500]}")

        return output_path


@app.post("/api/generate")
async def generate_video(
    video: UploadFile = File(...),
    style: str = Form(...),
):
    # Parse style JSON
    try:
        style_data = json.loads(style)
        if not isinstance(style_data, dict):
            raise ValueError("style must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid style JSON: {str(e)}")

    # Lazy import whisper if needed and load model
    global whisper, whisper_model
    if whisper is None:
        try:
            import whisper as _whisper  # type: ignore
            whisper = _whisper
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper not available: {str(e)}")

    if whisper_model is None:
        try:
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {str(e)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, video.filename or "input.mp4")
        try:
            with open(input_path, "wb") as f:
                f.write(await video.read())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save uploaded video: {str(e)}")

        try:
            output_path = subtitle_video_oneword(
                input_path,
                model=whisper_model,
                **style_data,
            )
        except TypeError as te:
            raise HTTPException(status_code=400, detail=f"Invalid style parameters: {str(te)}")

        try:
            with open(output_path, "rb") as f:
                data = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read output: {str(e)}")

    headers = {"Content-Disposition": "attachment; filename=final_video.mp4"}
    return Response(content=data, media_type="video/mp4", headers=headers)


@app.get("/")
def read_root():
    return {"message": "Video Captioning API ready"}


@app.get("/test")
def test_status():
    return {
        "backend": "running",
        "whisper_model": WHISPER_MODEL_SIZE,
        "model_loaded": whisper_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
