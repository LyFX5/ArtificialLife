

import interface_tools as itf
from data_base_api import DataBaseInterface
import media
from accompaniment_generation import AccompanimentGenerator


if __name__ == "__main__":

    generator = AccompanimentGenerator()
    data_base = DataBaseInterface()
    media_processor = media.MediaProcessor()

    command = input("Write a command, please !" + "\n")
    command = command.split()

    assert len(command) >= 2, "Such a command does not exist !"

    if command[0] == itf.UserCommand.LOAD_AUDIO_MEDIA.value: # "load_audio_media hhtps: . . . "

        audio_media = media.Media()
        hyperlink = command[1]
        audio_media.create_media_from_hyperlink(hyperlink, media.TypesOfMedia.AUDIO)

        data_base.upload_media(audio_media)

    elif command[0] == itf.UserCommand.LOAD_VIDEO_MEDIA.value: # "load_video_media hhtps: . . . "

        video_media = media.Media()
        hyperlink = command[1]
        video_media.create_media_from_hyperlink(hyperlink, media.TypesOfMedia.VIDEO)

        data_base.upload_media(video_media)

    elif command[0] == itf.UserCommand.LOAD_VIDEO_WITH_AUDIO_MEDIA.value: # "load_video_with_audio_media hhtps: . . . "

        video_with_audio_media = media.Media()
        hyperlink = command[1]
        video_with_audio_media.create_media_from_hyperlink(hyperlink, media.TypesOfMedia.VIDEO_WITH_AUDIO)

        data_base.upload_media(video_with_audio_media)

    elif command[0] == itf.UserCommand.GENERATE_ACCOMPANIMENT_FOR_MEDIA.value: # "generate_accompaniment_for_media (media ID or name)"

        media_ID_or_name = command[1]

        media = data_base.get_media(media_ID_or_name)

        accompaniment_media = generator.generate_accompaniment(media)

        data_base.upload_media(accompaniment_media)

    elif command[0] == itf.UserCommand.MERGE_VIDEO_MEDIA_WITH_AUDIO_MEDIA.value: # "merge_video_media_with_audio_media (video media ID or name) (audio media ID or name)"

        video_media_ID_or_name = command[1]
        audio_media_ID_or_name = command[2]

        video_media = data_base.get_media(video_media_ID_or_name)
        audio_media = data_base.get_media(audio_media_ID_or_name)

        video_media_with_audio_media = media_processor.merge_video_media_with_audio_media(video_media, audio_media)

        data_base.upload_media(video_media_with_audio_media)

    elif command[0] == itf.UserCommand.APPRECIATE_ACCOMPANIMENT_OF_THE_MEDIA.value: # "appreciate_accompaniment_of_the_media (media ID or name) (reaction)"

        media_ID_or_name = command[1]
        reaction = command[2]

        data_base.update_reaction(media_ID_or_name, reaction)

    else:
        print("Such a command does not exist !")


# load_audio_media https://www.youtube.com/watch?v=1hRTcSp0X1U
# load_video_with_audio_media https://www.youtube.com/watch?v=1hRTcSp0X1U


# Пользователь загружает картинку. Мы выдаем несколько вариантов совмещения этой картинки с другими и выделяем лучшее
# по нашему мнению слияние. Пользователь оценивает или не оценивает эти слияния.