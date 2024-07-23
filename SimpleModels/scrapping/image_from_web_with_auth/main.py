import telebot
import web_interaction as web

# davydovf12@gmail.com
# Qwerty1234

STORING_FOLDER = "images_storage"
bot = telebot.TeleBot("6619326384:AAHa-0PMtRwqFLT4rHMKBRAU3E9-WvUWTcY")


@bot.message_handler(commands=["start"])
def start(message):
    mess = f"Hello, <b>{message.from_user.first_name} <u>{message.from_user.last_name}</u></b>"
    bot.send_message(message.chat.id, mess, parse_mode="html")


@bot.message_handler(content_types=["text"])
def get_user_text(message):
    if len(message.text) > 0:
        text = message.text
        filename = f"{STORING_FOLDER}/image.jpg"
        web.download_image_by_text(text, filename)
        with open(filename, 'rb') as photo:
            bot.send_photo(message.chat.id, photo)


if __name__ == '__main__':
    bot.polling(none_stop=True)


