
import interface_tools as itf
from data_base_api import DataBaseInterface
import art
from accompaniment_generation import AccompanimentGenerator

if __name__ == "__main__":

    generator = AccompanimentGenerator()
    data_base = DataBaseInterface()

    command = input("Write a command, please !")
    command = command.split()

    assert len(command) >= 2, "Such a command does not exist !"

    if command[0] == itf.UserCommand.LOAD_AUDIO_ART.value:
        # "load_audio_art hhtps: . . . "

        audio_art = art.AudioArt()
        hyperlink = command[1]
        audio_art.create_audio_art_from_hyperlink(hyperlink)

        data_base.upload_art(audio_art)

    elif command[0] == itf.UserCommand.LOAD_VIDEO_ART.value:
        # "load_video_art hhtps: . . . "

        video_art = art.VideoArt()
        hyperlink = command[1]
        video_art.create_video_art_from_hyperlink(hyperlink)

        data_base.upload_art(video_art)

    elif command[0] == itf.UserCommand.GENERATE_ACCOMPANIMENT_FOR_ART.value:
        # "generate_accompaniment_for_art (art ID or name)"

        art_ID_or_name = command[1]

        art = data_base.get_art(art_ID_or_name)

        accompaniment_art = generator.generate_accompaniment(art)

        data_base.upload_art(accompaniment_art) # этот метод выводит название и ID загруженного произведения

    elif command[0] == itf.UserCommand.APPRECIATE_ACCOMPANIMENT_OF_THE_ART.value:
        # "appreciate_accompaniment_of_the_art (art ID or name) (reaction)"

        art_ID_or_name = command[1]
        reaction = command[2]

        data_base.update_reaction(art_ID_or_name, reaction)

    else:
        print("Such a command does not exist !")


