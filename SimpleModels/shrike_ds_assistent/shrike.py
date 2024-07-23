import telebot
from graphs_management_utilities import dict_of_lists_to_graph, pasts_to_tags_graph, posts_dict, tags_dict
from tags_management_utilities import tags_list, tags_string
from youtube_downloader import download_video
from audio_extract import extract_audio
import os


SYSTEMS_COMMUNICATION_CHANNEL_ID = "-1002078911910"
IMAGES_STORING_FOLDER = "images_storage"
YOUTUBE_VIDEOS_FOLDER = "youtube_videos_storage"
bot = telebot.TeleBot("7160809212:AAEA86R2N9ljmDlZSOvUsSExa9zEoetELbo")
add_tags_message_id = ""


def validate_message_is_id(message):
    return True


@bot.message_handler(commands=["start"])
def start(message):
    mess = f"Hello, <b>{message.from_user.first_name} <u>{message.from_user.last_name}</u></b>"
    bot.send_message(message.chat.id, mess, parse_mode="html")


@bot.message_handler(content_types=["text"])
def youtube_download(message):
    youtube_url = message.text
    video_title = download_video(youtube_url, YOUTUBE_VIDEOS_FOLDER, True)
    video_path = f"{YOUTUBE_VIDEOS_FOLDER}/{video_title}.mp4"
    audio_path = f"{YOUTUBE_VIDEOS_FOLDER}/{video_title}.mp3"
    extract_audio(input_path=video_path,
                  output_path=audio_path,
                  start_time="00:00",
                  overwrite=True)
    with open(audio_path, "rb") as video:
        bot.send_audio(message.chat.id, video)
    os.remove(video_path)
    os.remove(audio_path)


@bot.message_handler(commands=["add_tags"])
@bot.edited_message_handler(func=lambda message: True)
def add_tags(message):
    # TODO async
    chat_id = message.chat.id
    print(chat_id)
    print(message.id)
    bot.send_message(chat_id, "yes "+message.text)
    bot.edit_message_text(chat_id=chat_id, text="newtag", message_id=992)


@bot.message_handler(commands=["graphs"])
def build_graphs(message):
    posts_gra, triplets = dict_of_lists_to_graph(posts_dict, name="g_posts", filename="posts.gv",
                                                 label='Associative Graph of Posts')
    posts_gra_path = posts_gra.render(directory="graph_storage") # , format="jpg"
    tags_gra, triplets = dict_of_lists_to_graph(tags_dict, name="g_tags", filename="tags.gv",
                                                label='Associative Graph of Tags')
    tags_gra_path = tags_gra.render(directory="graph_storage")
    posts_to_tags_gra = pasts_to_tags_graph(posts_dict, name="dig_posts_tags", filename="posts_tags.gv",
                                            label='Posts to Tags Map')
    posts_to_tags_gra_path = posts_to_tags_gra.render(directory="graph_storage")
    with open(posts_gra_path, 'rb') as posts_gra_file:
        bot.send_document(message.chat.id, posts_gra_file)
    with open(tags_gra_path, 'rb') as tags_gra_file:
        bot.send_document(message.chat.id, tags_gra_file)
    with open(posts_to_tags_gra_path, 'rb') as posts_to_tags_gra_file:
        # SYSTEMS_COMMUNICATION_CHANNEL_ID
        bot.send_document(message.chat.id, posts_to_tags_gra_file)


if __name__ == '__main__':
    bot.polling(none_stop=True)

