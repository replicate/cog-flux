import asyncio
import tempfile

from cog import Path


async def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise ValueError(f"Failed to get video duration: {stderr.decode()}")

    try:
        return float(stdout.decode().strip())
    except ValueError as e:
        raise ValueError(f"Failed to parse video duration: {e}")


async def get_video_resolution(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise ValueError(f"Failed to get video resolution: {stderr.decode()}")

    try:
        width_str, height_str = stdout.decode().strip().split("x")
        return int(width_str), int(height_str)
    except Exception as e:
        raise ValueError(f"Failed to parse video resolution: {e}")


async def get_first_frame(video_path: Path) -> Path:
    return await get_frame(video_path, frame="first")


async def get_last_frame(video_path: Path) -> Path:
    return await get_frame(video_path, frame="last")


async def get_frame(video_path: Path, frame: str | int = "first") -> Path:
    """
    Extract a specific frame from a video and save it as a temporary image file.

    Args:
        video_path: Path to the input video file
        frame: 'first', 'last', or an integer frame index (0-based)

    Returns:
        Path to the temporary image file containing the extracted frame
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = temp_file.name

    if frame == "first":
        # Extract the first frame
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            temp_file_path,
        ]
    elif frame == "last":
        # Get total frame count using ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
        probe_proc = await asyncio.create_subprocess_exec(
            *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        probe_out, probe_err = await probe_proc.communicate()
        if probe_proc.returncode != 0:
            raise ValueError(f"Failed to get frame count: {probe_err.decode()}")
        try:
            total_frames = int(probe_out.decode().strip())
        except Exception as e:
            raise ValueError(f"Failed to parse frame count: {e}")
        last_frame_idx = max(0, total_frames - 1)
        # Extract the last frame
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{last_frame_idx})",
            "-vsync",
            "0",
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            temp_file_path,
        ]
    elif isinstance(frame, int):
        # Extract a specific frame by index
        if frame < 0:
            raise ValueError("Frame index must be non-negative")
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-vsync",
            "0",
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            temp_file_path,
        ]
    else:
        raise ValueError("frame must be 'first', 'last', or a non-negative integer")

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise ValueError(f"Failed to extract frame: {stderr.decode()}")

    return Path(temp_file_path)
