

from pytube import YouTube
from enum import Enum



class TypesOfMedia(Enum):

    AUDIO = "audio"
    VIDEO = "video"
    VIDEO_WITH_AUDIO = "video_with_audio"


def to_ascii(text):
    ascii_str = ""
    for i in range(min(7, len(text))): # TODO так могут возникнуть коллизии
        ascii_str += str(ord(text[i]))
    return int(ascii_str)



class Media:

    def __init__(self):

        self.ID = 0
        self.source = None
        self.score = 0.
        self.tags = ""
        self.type = None
        self.folder_with_medias_path = "videos"
        self.title = None


    def create_audio_media_from_hyperlink(self, hyperlink: str):

        assert "https://www.youtube.com/" in hyperlink, "Provide YouTube video link, please!"

        self.source = hyperlink

        self.type = TypesOfMedia.AUDIO.value

        youtube_video = YouTube(hyperlink)

        self.title = youtube_video.title

        self.ID = to_ascii("a" + youtube_video.video_id)

        audio = youtube_video.streams.filter(only_audio=True)[0]

        audio.download(self.folder_with_medias_path, filename=f"{self.type}_{str(self.ID)}_{self.title}.mp4")

        print("Audio is downloaded.")


    def create_video_media_from_hyperlink(self, hyperlink: str):

        assert "https://www.youtube.com/" in hyperlink, "Provide YouTube video link, please!"

        self.source = hyperlink

        self.type = TypesOfMedia.VIDEO.value

        youtube_video = YouTube(hyperlink)

        self.title = youtube_video.title

        self.ID = to_ascii("v" + youtube_video.video_id)

        video = youtube_video.streams.filter(only_video=True)[0]

        video.download(self.folder_with_medias_path, filename=f"{self.type}_{str(self.ID)}_{self.title}.mp4")

        print("Video is downloaded.")


    def create_video_with_audio_media_from_hyperlink(self, hyperlink: str):

        assert "https://www.youtube.com/" in hyperlink, "Provide YouTube video link, please!"

        self.source = hyperlink

        self.type = TypesOfMedia.VIDEO_WITH_AUDIO.value

        youtube_video = YouTube(hyperlink)

        self.title = youtube_video.title

        self.ID = to_ascii("va" + youtube_video.video_id)

        video_with_audio = youtube_video.streams.get_highest_resolution()

        video_with_audio.download(self.folder_with_medias_path, filename=f"{self.type}_{str(self.ID)}_{self.title}.mp4")

        print("Video with audio is downloaded.")


class MediaProcessor:

    def __int__(self):

        self.var = 0


    def merge_video_media_with_audio_media(self, video_media: Media, audio_media: Media) -> Media:

        video_media_with_audio_media = None

        return video_media_with_audio_media








