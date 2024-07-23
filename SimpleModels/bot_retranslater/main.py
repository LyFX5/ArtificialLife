import telebot
from PIL import Image


CHANNEL_ID = "-1002018514287"
STORING_FOLDER = "images_storage"
bot = telebot.TeleBot("7160809212:AAEA86R2N9ljmDlZSOvUsSExa9zEoetELbo")


# Function to clear Metadata from a specified image.
def clear_all_metadata(imgname):
    # Open the image file
    img = Image.open(imgname)
    # Read the image data, excluding metadata.
    data = list(img.getdata())
    # Create a new image with the same mode and size but without metadata.
    img_without_metadata = Image.new(img.mode, img.size)
    img_without_metadata.putdata(data)
    # Save the new image over the original file, effectively removing metadata.
    img_without_metadata.save(imgname)
    return f"Metadata successfully cleared from '{imgname}'."


@bot.message_handler(commands=["start"])
def start(message):
    mess = f"Hello, <b>{message.from_user.first_name} <u>{message.from_user.last_name}</u></b>"
    bot.send_message(message.chat.id, mess, parse_mode="html")

@bot.message_handler(content_types=["text"])
def get_user_text(message):
    if len(message.text) > 0:
        text = message.text
        bot.send_message(message.chat.id, text)
        # filename = f"{STORING_FOLDER}/image.jpg"
        # web.download_image_by_text(text, filename)
        # with open(filename, 'rb') as photo:
        #     bot.send_photo(message.chat.id, photo)

@bot.message_handler(content_types=['photo'])
def photo(message):
    print(f"{message.photo=}")
    fileID = message.photo[-1].file_id
    print(f"{fileID=}")
    file_info = bot.get_file(fileID)
    print(f"{file_info.file_path=}")
    downloaded_file = bot.download_file(file_info.file_path)
    filename = f"{STORING_FOLDER}/image.jpg"
    with open(filename, 'wb') as photo_file:
        photo_file.write(downloaded_file)
    res_str = clear_all_metadata(filename)
    print(res_str)
    bot.send_message(message.chat.id, res_str)
    with open(filename, 'rb') as photo_file:
        bot.send_photo(CHANNEL_ID, photo_file)


if __name__ == '__main__':
    bot.polling(none_stop=True)


