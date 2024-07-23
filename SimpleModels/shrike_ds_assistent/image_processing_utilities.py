from PIL import Image


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