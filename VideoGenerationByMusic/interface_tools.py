
from enum import Enum

class UserCommand(Enum):

    LOAD_AUDIO_MEDIA = "load_audio_media"
    LOAD_VIDEO_MEDIA = "load_video_media"
    LOAD_VIDEO_WITH_AUDIO_MEDIA = "load_video_with_audio_media"

    GENERATE_ACCOMPANIMENT_FOR_MEDIA = "generate_accompaniment_for_media"

    APPRECIATE_ACCOMPANIMENT_OF_THE_MEDIA = "appreciate_accompaniment_of_the_media"

    MERGE_VIDEO_MEDIA_WITH_AUDIO_MEDIA = "merge_video_media_with_audio_media"
