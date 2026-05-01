import os
import soundfile as sf
from datetime import datetime

from config import RECORDINGS_DIR


class RecorderManager:
    """Manages recorded audio files."""

    def __init__(self, recordings_dir: str = RECORDINGS_DIR):
        self.recordings_dir = os.path.abspath(recordings_dir)
        os.makedirs(self.recordings_dir, exist_ok=True)

    def list_recordings(self) -> list[dict]:
        """Return all saved recordings."""
        recordings = []
        if not os.path.isdir(self.recordings_dir):
            return recordings
        
        for filename in sorted(os.listdir(self.recordings_dir), reverse=True):
            if filename.endswith(".wav"):
                path = os.path.join(self.recordings_dir, filename)
                try:
                    info = sf.info(path)
                    recordings.append({
                        "filename": filename,
                        "path": os.path.abspath(path),
                        "duration": info.duration,
                        "duration_str": f"{info.duration:.1f}s",
                        "created": datetime.fromtimestamp(
                            os.path.getmtime(path)
                        ).strftime("%Y-%m-%d %H:%M"),
                    })
                except Exception:
                    pass
        
        return recordings

    def dropdown_choices(self) -> list[str]:
        """Return recordings as dropdown choices."""
        recordings = self.list_recordings()
        if not recordings:
            return ["(无录音)"]
        return [
            f"{r['filename']} ({r['duration_str']}, {r['created']})"
            for r in recordings
        ]

    def parse_filename_from_choice(self, choice: str) -> str:
        """Extract filename from dropdown choice string."""
        if not choice or choice == "(无录音)":
            return None
        return choice.split(" (")[0]

    def table_data(self) -> list[list]:
        """Return recordings as table rows for Gradio dataframe."""
        recordings = self.list_recordings()
        if not recordings:
            return [["(无录音)", "", "", ""]]
        return [
            [r["filename"], r["duration_str"], r["created"], r["path"]]
            for r in recordings
        ]

    def save_recording(self, audio_tuple, name: str = None) -> str:
        """Save recorded audio. Returns absolute path."""
        if audio_tuple is None:
            return None
        
        sr, audio = audio_tuple
        
        if name and name.strip():
            safe_name = self._sanitize_name(name.strip())
            filename = f"{safe_name}.wav"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        
        path = os.path.join(self.recordings_dir, filename)
        sf.write(path, audio, sr)
        return os.path.abspath(path)

    def delete_recording(self, filename: str) -> bool:
        """Delete a recording file."""
        if not filename or filename == "(无录音)":
            return False
        path = os.path.join(self.recordings_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def get_recording_path(self, filename: str) -> str:
        """Get absolute path to a recording file."""
        if not filename or filename == "(无录音)":
            return None
        path = os.path.join(self.recordings_dir, filename)
        if os.path.isfile(path):
            return os.path.abspath(path)
        return None

    def get_recording_info(self, filename: str) -> dict:
        """Get recording info by filename."""
        if not filename or filename == "(无录音)":
            return None
        path = os.path.join(self.recordings_dir, filename)
        if os.path.isfile(path):
            try:
                info = sf.info(path)
                return {
                    "filename": filename,
                    "path": os.path.abspath(path),
                    "duration": info.duration,
                    "duration_str": f"{info.duration:.1f}s",
                }
            except Exception:
                pass
        return None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for filename."""
        safe = "".join(c if c.isalnum() or c in "_- " else "" for c in name)
        return safe.strip().replace(" ", "_") or "recording"