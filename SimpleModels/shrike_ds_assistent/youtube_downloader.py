from pytube import YouTube


def download_video(url, download_path, only_audio):
    get_video = YouTube(url)
    # only video or only audio
    media = get_video.streams.filter(only_audio=only_audio)[0]
    title = media.download(download_path)
    # audio and video in worst quality
    # get_stream = get_video.streams.first()
    # get_stream.download(folder)
    # audio and video in the best quality
    # get_stream = get_video.streams.get_highest_resolution()
    # get_stream.download(folder)
    return title.split("/"+download_path)[-1][1:].replace(".mp4", "")
