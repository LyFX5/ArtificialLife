
from art import Art
from enum import Enum
import random

class Column(Enum):
    ID = 'ID'
    DURATION = 'duration'
    CONTENT = 'content'
    StartOfParagraf = 'startOfParagraph'
    StartTime = 'startTime'
    LinkToVideo = 'linkToVideo'

import sqlite3 as sql
from tqdm import tqdm

class TableManager:
    def __init__(self):
        self.db_file_name = "subtitles.db"
        self.table_name = 'subtitles'
        self.cursor = None
        self.connection = None

    def create_connection(self):
        self.connection = sql.connect(self.db_file_name)

        # def dict_factory(cursor, row):
        #     # обертка для преобразования
        #     # полученной строки. (взята из документации)
        #     d = {}
        #     for idx, col in enumerate(cursor.description):
        #         d[col[0]] = row[idx]
        #     return d
        #
        # self.connection.row_factory = dict_factory

    def create_cursor(self):
        assert self.connection != None, 'connection is not created'
        self.cursor = self.connection.cursor()

    def table_not_exists(self) -> bool:
        assert self.cursor != None, 'cursor is not created'
        self.cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
        db = self.cursor.fetchall()
        print(f'existing tables {db}')
        return len(db) == 0 or self.table_name not in db[0] # [table['name'] for table in db]

    def create_table(self):
        if self.table_not_exists():

            query = f'''CREATE TABLE {self.table_name} 
                        ({Column.ID.value} integer, {Column.LinkToVideo.value} text, {Column.DURATION.value} integer, {Column.CONTENT.value} text, {Column.StartOfParagraf.value} integer, {Column.StartTime.value} integer)'''
            self.cursor.execute(query)

        print(f'Table {self.table_name} is created.')

    def save_changes(self):
        # Save (commit) the changes
        self.connection.commit()

    def close_connection(self):

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        self.connection.close()

    def upload_video_to_table(self, list_of_rows: list):
        # the list_of_rows must be a list of tuples (is a subtitles of the video)

        ID = list_of_rows[0][0]
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE {Column.ID.value}={ID} LIMIT 1)"
        res = self.cursor.execute(query).fetchall()
        #print(res)
        id_not_exists = res[0][0] == 0

        if id_not_exists:
            query = f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?)"
            self.cursor.executemany(query, list_of_rows)

    def upload_videos_to_table(self, videos):

        for list_of_rows in tqdm(videos):
            self.upload_video_to_table(list_of_rows)

        print(f'All {len(videos)} videos were uploaded to the database.')

    def find_in_table(self, column: Column, value):

        query = f'''
                 SELECT * FROM {self.table_name}
                 WHERE {column.value} like '{value}'                
                 '''

        return self.cursor.execute(query).fetchall()

    def find_string_usage_in_subtitles(self, string: str):

        try:
            results = self.find_in_table(column=Column.CONTENT, value=f'% {string} %')
        except Exception as e: #TODO
            return None

        if len(results) == 0:
            return None
        elif len(results[0]) == 0:
            return None
        else:
            return results


class DataBaseInterface:

    def __init__(self):

        self.db_file_path = "db_file_path"


    def upload_art(self, art: Art):
        # этот метод выводит название и ID загруженного произведения
        pass


    def get_art(self, art_ID_or_name):

        pass


    def update_reaction(self, art_ID_or_name, reaction):

        pass








