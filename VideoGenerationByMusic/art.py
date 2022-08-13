
class Art:

    def __init__(self):

        self.ID = 0
        self.source = None
        self.score = 0
        self.tags = []

        self.type = None


class AudioArt(Art):

    def __init__(self):
        super(AudioArt, self).__init__()

        self.format = None


    def create_audio_art_from_hyperlink(self, hyperlink: str):

        pass


class VideoArt(Art):

    def __init__(self):
        super(VideoArt, self).__init__()

        self.format = None


    def create_video_art_from_hyperlink(self, hyperlink: str):

        pass


