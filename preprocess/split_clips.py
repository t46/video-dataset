import os
import json
import imageio
import numpy as np
from streaming import MDSWriter
from tqdm import tqdm
import ffmpeg

BASE_DIR = "/root/projects/rl-nlp/videos/dataset/00000/"

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

videos = []

for video_filename in tqdm(os.listdir(BASE_DIR)):
    if not video_filename.endswith(".mp4"):
        continue

    file_basename = os.path.splitext(video_filename)[0]
    video_file_path = os.path.join(BASE_DIR, video_filename)
    json_file_path = os.path.join(BASE_DIR, f"{file_basename}.json")

    if os.path.exists(json_file_path):
        pass
    else:
        continue

    reader = imageio.get_reader(video_file_path)

    # probe = ffmpeg.probe(video_file_path)
    # video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    # print(video_info)

    # probe = ffmpeg.probe(video_file_path, select_streams="v:0", show_entries="frame=pkt_pts_time,key_frame", read_intervals="%+#1")
    # frames = probe["frames"]
    # keyframe_timestamps = [frame["pkt_pts_time"] for frame in frames if frame["key_frame"] == 1]
    # print(keyframe_timestamps)

    frames = []
    subtitles = []

    with open(json_file_path, 'r') as f:
        data = json.load(f)
        fps = reader.get_meta_data()['fps']
        # fps = data["yt_meta_dict"]["info"]["fps"]
        try:
            subtitle_data = data["yt_meta_dict"]["subtitles"]
        except KeyError:
            continue
        # if data["yt_meta_dict"] is None:
        #     continue

        for i, subtitle in enumerate(subtitle_data):
            start_time = convert_time_to_seconds(subtitle["start"])
            end_time = convert_time_to_seconds(subtitle["end"])

            j = 0
            # current_time = 0.0
            while start_time < end_time:
                chunk_end_time = start_time + 1.0  # 60 seconds
                if chunk_end_time > end_time:
                    chunk_end_time = end_time
                    # break  # Because we get error around the ened of the video
                
                # while current_time < chunk_end_time:
                frame_idx = int(start_time * fps)
                # reader = imageio.get_reader(video_file_path)  # Initialize here to avoid potential errors caused by jumping between frames

                try: 
                    frame = reader.get_data(frame_idx)
                except IndexError:
                    break

                if frame is None:
                    break
                frame_array = np.array(frame)  # Convert each frame to a numpy array

                frames.append(frame_array)
                subtitles.append(np.frombuffer(subtitle["lines"][0].encode('utf-8'), dtype=np.uint8))

                start_time = chunk_end_time
                j += 1

        # cap.release()
    try:
        MAX_LENGTH = max([len(sub) for sub in subtitles])
    except ValueError:
        print('max() arg is an empty sequence')
        continue

    padded_subtitles = [np.pad(sub, (0, MAX_LENGTH - len(sub))) for sub in subtitles]

    video_data = {
        'frame': np.asarray(frames, dtype=np.float32),
        'subtitle': np.asarray(padded_subtitles)
    }

    videos.append(video_data)

SAVE_PATH = "/root/projects/rl-nlp/videos/data/mds/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle': 'ndarray'
}
COMPRESSION = 'zstd'
HASHES = 'sha1', 'xxh64'

with MDSWriter(out=SAVE_PATH, columns=COLUMNS, compression=COMPRESSION, hashes=HASHES) as save_file:
    for video_data in videos:
        save_file.write(video_data)
