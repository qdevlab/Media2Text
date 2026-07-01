import argparse
import json
import os
import subprocess
import pathlib
import time
import shutil
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ===================================================================== #
#  Settings
# ===================================================================== #

# .env next to the script: OPENAI_API_KEY (required), FFMPEG_PATH/FFPROBE_PATH (optional)
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

MP4_SIGN_DONE = "+"

# ffmpeg/ffprobe are taken from PATH by default; override in .env if they are not there
# (Windows example: FFMPEG_PATH=c:\tools\ffmpeg\ffmpeg.exe)
FFMPEG_PATH  = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
FFMPEG_SEGMENT_TIME = 300
FFMPEG_SEGMENT_DELTA = 8
FFMPEG_SEGMENT_MASK_BODY = "output"
FFMPEG_SEGMENT_MASK = FFMPEG_SEGMENT_MASK_BODY + "_%04d.mp4"

TC_MIN_GAP = 30.0          # min seconds between timecodes (0 = every segment)
SHOW_SPEAKER = False       # print speaker label

RESULT_SUFFIX = "__text"   # result files: <orig>__text.txt / <orig>__text.timecodes.txt

LANGUAGE = 'en'

# ===================================================================== #
#  OpenAI client
# ===================================================================== #

API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("[!] OPENAI_API_KEY not found. Set it in .env: OPENAI_API_KEY=sk-...")

g_client = OpenAI(api_key=API_KEY)


# ===================================================================== #
#  Timecodes
# ===================================================================== #

def seconds_to_timecode(seconds: float) -> str:
    if seconds is None:
        return "??:??:??"
    total = int(round(float(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_segments_with_timecodes(
    segments: list,
    time_offset: float = 0.0,
    min_gap: float = TC_MIN_GAP,
    show_speaker: bool = SHOW_SPEAKER,
) -> str:
    # Consecutive lines from one speaker are merged; timecode printed no more than once per min_gap.
    # time_offset is the part offset: ffmpeg -reset_timestamps 1 zeroes each segment's timings.
    lines = []
    prev_speaker = None
    last_tc_time = None
    buffer = []

    def flush():
        if buffer and last_tc_time is not None:
            tc = seconds_to_timecode(last_tc_time)
            body = " ".join(buffer)
            if show_speaker:
                lines.append(f"[{tc}] {prev_speaker}: {body}")
            else:
                lines.append(f"[{tc}] {body}")

    for seg in segments:
        start   = (seg.get("start") or 0.0) + time_offset
        speaker = seg.get("speaker") or "?"
        text    = (seg.get("text") or "").strip()
        if not text:
            continue

        start_new = (
            speaker != prev_speaker
            or last_tc_time is None
            or min_gap == 0
            or (start - last_tc_time) >= min_gap
        )

        if start_new:
            flush()
            buffer = [text]
            prev_speaker = speaker
            last_tc_time = start
        else:
            buffer.append(text)

    flush()
    return "\n".join(lines)


def ffprobe_duration(path) -> float:
    try:
        out = subprocess.run(
            [
                FFPROBE_PATH,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return float(FFMPEG_SEGMENT_TIME)


def get_segment_duration(result: dict, mp4_file) -> float:
    # duration from API -> end of last segment -> ffprobe
    dur = result.get("duration")
    if dur:
        return float(dur)

    segs = result.get("segments") or []
    if segs and segs[-1].get("end"):
        return float(segs[-1]["end"])

    return ffprobe_duration(mp4_file)


# ===================================================================== #
#  Latin-drift detector (the model sometimes slips into English even with language='ru')
# ===================================================================== #

def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    cyr = sum(1 for c in letters if ('а' <= c.lower() <= 'я') or c.lower() == 'ё')
    return cyr / len(letters)


def flag_language_flips(segments: list, time_offset: float = 0.0,
                        min_cyrillic: float = 0.5, min_len: int = 10) -> list:
    # Segments with a low share of Cyrillic (possible drift to English). Short ones are skipped.
    flips = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if len(text) < min_len:
            continue
        if cyrillic_ratio(text) < min_cyrillic:
            start = (seg.get("start") or 0.0) + time_offset
            flips.append((seconds_to_timecode(start), text))
    return flips


# ===================================================================== #

def rename_to_done(path):
    # Prefix the processed mp4 with '+'. Tolerant of a locked file on Windows (several retries).
    op = Path(path)
    target = op.with_name(f"{MP4_SIGN_DONE}{op.name}")

    if op.name.startswith(MP4_SIGN_DONE):
        return
    if not op.exists():
        print(f"[!] File not found for rename: {op.name}")
        return
    if target.exists():
        print(f"[!] Target already exists, rename skipped: {target.name}")
        return

    last_err = None
    for _ in range(8):
        try:
            op.replace(target)
            print(f"[+] Renamed -> {target.name}")
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.5)
    print(f"[!] Could not rename {op.name} (locked?): {last_err}")


def transcribe_diarize(
    client: OpenAI,
    audio_path: str,
    language: str = "ru",
    stream: bool = False,
) -> dict:
    # -> dict: text, segments, duration
    with open(audio_path, "rb") as audio_file:
        params = {
            "model": "gpt-4o-transcribe-diarize",
            "file": audio_file,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        if language:
            params["language"] = language

        output = {}

        if stream:
            params["stream"] = True
            segments = []
            full_text = ""
            duration = None

            response = client.audio.transcriptions.create(**params)
            for event in response:
                if event.type == "transcript.text.segment":
                    seg = {
                        "id": event.id,
                        "speaker": event.speaker,
                        "start": event.start,
                        "end": event.end,
                        "text": event.text,
                    }
                    segments.append(seg)
                elif event.type == "transcript.text.done":
                    full_text = event.text

            if segments and segments[-1].get("end"):
                duration = segments[-1]["end"]

            output = {
                "text": full_text,
                "segments": segments,
                "duration": duration,
            }
            return output

        else:
            response = client.audio.transcriptions.create(**params)

            text = response.text

            segments = [
                {
                    "id": seg.id,
                    "speaker": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in (response.segments or [])
            ]

            duration = getattr(response, "duration", None)

            output = {
                "text": text,
                "segments": segments,
                "duration": duration,
            }
            return output


def write_text_by_mp4name(mp4file, text, suffix=""):
    op = Path(mp4file)
    out = op.parent / f"{op.stem}{suffix}.txt"
    with open(out, 'w', encoding='utf-8') as file:
        file.write(text)


def process_parts(parts_folder, language):

    folder = pathlib.Path(parts_folder)

    # Sort by name: rglob order is not guaranteed, otherwise the offset drifts and timecodes break.
    mp4_files = sorted(
        folder.rglob(FFMPEG_SEGMENT_MASK_BODY + '*.mp4'),
        key=lambda p: p.name,
    )
    for mp4_file in mp4_files:
        print(f"\t[*] segment file : {mp4_file}")

    text_summ    = ""
    text_summ_tc = ""
    time_offset  = 0.0

    for mp4_file in mp4_files:

        start_time = time.time()

        # Audio api
        result = transcribe_diarize(g_client, str(mp4_file), language=language)

        tc_text = format_segments_with_timecodes(result["segments"], time_offset)

        flips = flag_language_flips(result["segments"], time_offset)
        if flips:
            print(f"\t[!] Possible language flip in {len(flips)} segment(s) (check manually):")
            for tc, txt in flips[:10]:
                print(f"\t    [{tc}] {txt[:70]}")
            if len(flips) > 10:
                print(f"\t    ... and {len(flips) - 10} more")

        print(f"\t[+] {mp4_file.name} : {time.time() - start_time:.1f}s")

        text_summ    += "\n\n\n" + result["text"]
        text_summ_tc += "\n\n\n" + tc_text

        time_offset += get_segment_duration(result, mp4_file)

    return text_summ, text_summ_tc


# ===================================================================== #

def main(media_folder, language=None):

    folder = pathlib.Path(media_folder)

    # Recursive search
    mp4_files = []
    for mp4_file in folder.rglob('*.mp4'):

        mp4_file_name = mp4_file.stem

        if (mp4_file_name.startswith(FFMPEG_SEGMENT_MASK_BODY)):
            continue
        if (mp4_file_name.startswith(MP4_SIGN_DONE)):
            continue

        print(f"[*] mp4 file : {str(mp4_file)}")
        mp4_files.append(mp4_file)

    # Enum
    for mp4_file in mp4_files:
        try:
            mp4_folder = pathlib.Path(mp4_file).parent
            print(f"[*] Folder : {mp4_folder}")

            # Per-file random working folder for the split parts
            work_dir = Path(tempfile.mkdtemp(prefix="m2t_", dir=str(mp4_folder)))
            try:
                mp4_output_mask = str(work_dir / FFMPEG_SEGMENT_MASK)

                # Split MP4
                sp = subprocess.run(
                    [
                        FFMPEG_PATH,
                        "-i", str(mp4_file),
                        "-vn",
                        "-acodec", "copy",
                        "-f", "segment",
                        "-segment_time", str(FFMPEG_SEGMENT_TIME),
                        "-segment_time_delta", str(FFMPEG_SEGMENT_DELTA),
                        "-reset_timestamps", "1",
                        mp4_output_mask,
                    ]
                )

                print(f"[*] FFMpeg exit code : {sp.returncode}")
                print(f"[*] FFMpeg stdout    : {sp.stdout}")

                all_text, all_text_tc = process_parts(str(work_dir), language=language)

                # Result files next to the original: <orig>__text.txt and <orig>__text.timecodes.txt
                write_text_by_mp4name(mp4_file, all_text, RESULT_SUFFIX)
                write_text_by_mp4name(mp4_file, all_text_tc, RESULT_SUFFIX + ".timecodes")

                rename_to_done(str(mp4_file))
            finally:
                # Drop the random working folder with all its parts
                shutil.rmtree(work_dir, ignore_errors=True)

        except Exception as e:
            # one bad file must not bring down the whole batch
            print(f"[!] Error on file {mp4_file}: {e}")
            continue

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio/video into text.")
    parser.add_argument("folder", help="Folder scanned recursively for .mp4 files")
    parser.add_argument("--language", default=LANGUAGE, help="Transcription language code, e.g. 'ru', 'en'")
    args = parser.parse_args()

    print("[*] Start")
    main(args.folder, language=args.language)
    print("[*] Done")
