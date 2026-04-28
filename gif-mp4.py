import os
from moviepy.editor import VideoFileClip

def convert_gifs_to_mp4():
    folder = os.getcwd()
    gifs = [f for f in os.listdir(folder) if f.lower().endswith(".gif")]

    if not gifs:
        print("No GIF files found in this folder.")
        return

    for gif in gifs:
        mp4_name = os.path.splitext(gif)[0] + ".mp4"
        mp4_path = os.path.join(folder, mp4_name)

        if os.path.exists(mp4_path):
            print(f"Skipping {gif} — {mp4_name} already exists.")
            continue

        print(f"Converting {gif} → {mp4_name}...")
        try:
            clip = VideoFileClip(os.path.join(folder, gif))
            clip.write_videofile(mp4_path, logger=None)
            clip.close()
            print(f"Done: {mp4_name}")
        except Exception as e:
            print(f"Failed to convert {gif}: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    convert_gifs_to_mp4()