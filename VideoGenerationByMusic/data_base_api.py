
from media import Media
from enum import Enum
import random
import sqlite3 as sql
from tqdm import tqdm



class Column(Enum):

    ID = 'ID'
    TYPE = "type"
    TITLE = "title"
    SCORE = "score"
    SOURCE = "source"
    FOLDER = "folder"
    TAGS = "tags"



class TableManager:

    def __init__(self):

        self.db_file_name = "media_data_base.db"
        self.table_name = 'media_table'
        self.cursor = None
        self.connection = None


    def create_connection(self):

        self.connection = sql.connect(self.db_file_name)


    def create_cursor(self):

        assert self.connection != None, 'Connection is not created!'

        self.cursor = self.connection.cursor()


    def table_not_exists(self) -> bool:

        assert self.cursor != None, 'Cursor is not created!'

        self.cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
        db = self.cursor.fetchall()
        print(f'existing tables {db}')

        return len(db) == 0 or self.table_name not in db[0] # [table['name'] for table in db]


    def create_table(self):

        if self.table_not_exists():

            query = f'''CREATE TABLE {self.table_name} 
                        ({Column.ID.value} integer, {Column.TYPE.value} text, {Column.TITLE.value} text, {Column.SCORE.value} float, {Column.SOURCE.value} text, {Column.FOLDER.value} text, {Column.TAGS.value} text)'''
            self.cursor.execute(query)

        print(f'Table {self.table_name} is created.')


    def save_changes(self):
        # Save (commit) the changes
        self.connection.commit()


    def close_connection(self):

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        self.connection.close()


    def upload_row_to_table(self, row: tuple):

        ID = row[0]
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE {Column.ID.value}={ID} LIMIT 1)"
        res = self.cursor.execute(query).fetchall()
        #print(res)
        id_not_exists = res[0][0] == 0

        if id_not_exists:
            query = f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?, ?)"
            self.cursor.execute(query, row)

            print('Row was uploaded to the database.')
            return True

        print('Row is already exist.')
        return False


    def find_in_table(self, column: Column, value):

        query = f'''
                 SELECT * FROM {self.table_name}
                 WHERE {column.value} like '{value}'                
                 '''

        return self.cursor.execute(query).fetchall()



class DataBaseInterface:

    def __init__(self):

        self.db_file_name = "media_data_base.db"
        self.table_name = 'media_table'


        self.table_manager = TableManager()

        self.table_manager.create_connection()
        self.table_manager.create_cursor()

        self.table_manager.create_table()

        self.table_manager.save_changes()
        self.table_manager.close_connection()


    def upload_media(self, media: Media):

        self.table_manager.create_connection()
        self.table_manager.create_cursor()

        row = (media.ID, media.type, media.title, media.score, media.source, media.folder_with_medias_path, media.tags)
        self.table_manager.upload_row_to_table(row)
        print(f"Media {media.ID} with title \"{media.title}\" is uploaded to the media table.")

        self.table_manager.save_changes()
        self.table_manager.close_connection()


    def get_media(self, media_ID_or_name):

        pass


    def update_reaction(self, media_ID_or_name, reaction):

        pass








