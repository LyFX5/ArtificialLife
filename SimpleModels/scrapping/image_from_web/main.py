import telebot
import web_interaction as web


STORING_FOLDER = "images_storage"
memory = []
bot = telebot.TeleBot("6619326384:AAHa-0PMtRwqFLT4rHMKBRAU3E9-WvUWTcY")


@bot.message_handler(commands=["start"])
def start(message):
    mess = f"Hello, <b>{message.from_user.first_name} <u>{message.from_user.last_name}</u></b>"
    bot.send_message(message.chat.id, mess, parse_mode="html")


@bot.message_handler(content_types=["text"])
def get_user_text(message):
    if len(message.text) > 0:
        memory.append(message.text)
        print(memory)
    word = message.text
    uuid = web.make_post(word)
    response_jsn = web.make_get(uuid)
    while response_jsn["images"] is None:
        response_jsn = web.make_get(uuid)
    image_base64 = response_jsn["images"][0]
    imagedata = web.decode_base64_to_image(image_base64)
    filename = f"{STORING_FOLDER}/image.jpg"
    with open(filename, 'wb') as f:
        f.write(imagedata)
    with open(filename, 'rb') as photo:
        bot.send_photo(message.chat.id, photo)


if __name__ == '__main__':
    bot.polling(none_stop=True)


