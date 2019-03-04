rem https://gist.github.com/protrolium/e0dbd4bb0f1a396fcb55
@echo off
ffmpeg -i part_43.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 2 part_43.wav