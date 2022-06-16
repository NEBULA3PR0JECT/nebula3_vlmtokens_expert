from abc import ABC, abstractmethod
from asyncio import constants
from typing import Optional
from fastapi import FastAPI
from threading import Lock
import logging
import os
import sys
import urllib
from typing import List
# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


from common.defines import *
from common.models import ExpertParam
from common.constants import OUTPUT

from nebula3_experts.nebula3_pipeline.nebula3_database.movie_db import MOVIE_DB
from nebula3_experts.nebula3_pipeline.nebula3_database.movie_s3 import MOVIE_S3
from nebula3_experts.nebula3_pipeline.nebula3_database.movie_tokens import MovieTokens, TokenEntry
from nebula3_experts.nebula3_pipeline.nebula3_database.config import NEBULA_CONF


DEFAULT_FILE_PATH = "/tmp/file.mp4"

class BaseExpert(ABC):
    def __init__(self):
        self.db_conf = NEBULA_CONF()
        self.movie_db = MOVIE_DB()
        self.db = self.movie_db.db
        self.movie_s3 = MOVIE_S3()
        self.movie_tokens = MovieTokens(self.db)
        self.status = ExpertStatus.STARTING
        self.tasks_lock = Lock()
        self.tasks = dict()
        self.temp_file = DEFAULT_FILE_PATH
        self.url_prefix = self.db_conf.get_webserver()

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def log_debug(self, msg):
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log_info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def log_error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print(msg)

    def set_active(self):
        """ setting the expert status to active"""
        self.status = ExpertStatus.ACTIVE

    def add_task(self, task_id: str,  taks_params = dict()):
        with self.tasks_lock:
            self.tasks[task_id] = taks_params

    def remove_task(self, task_id: str):
        with self.tasks_lock:
            self.tasks.pop(task_id)

    @abstractmethod
    def add_expert_apis(self, app: FastAPI):
        """add expert's specific apis (REST)

        Args:
            app (FastAPI): _description_
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """return expert's name
        """
        pass

    # @abstractmethod
    def get_status(self) -> str:
        """return expert's status
        """
        return self.status.name

    def get_cfg(self) -> dict:
        """return expert's config params
        """
        return {}

    # @abstractmethod
    def get_tasks(self) -> list:
        """ return the taks currently running """
        current_tasks = list()
        with self.tasks_lock:
            for id, info in self.tasks.items:
                current_tasks.append({ 'id': id, 'info': info })
        return current_tasks

    def get_dependency(self) -> str:
        """return the expert's dependency in the pipeline:
        which pipeline step is this expert depends on
        pass

        Returns:
            str: _description_
        """
        pass

    # @abstractmethod
    def handle_msg(self, msg_params):
        """handling msg: going over the movies and calling predict on each one
        Args:
            msg_params (_type_): _description_
        """
        output = OutputStyle.DB if not msg_params[constants.OUTPUT] else msg_params[constants.OUTPUT]
        movies = msg_params[constants.MOVIES]
        if isinstance(movies, list):
            for movie_id in movies:
                self.predict(movie_id, output)

    @abstractmethod
    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        pass


    # @abstractmethod
    def handle_exit(self):
        """handle things before exit process
        """
        print(f'Exiting from: {self.get_name()}')

    # utilities
    def download_video_file(self, movie_id, file_location = None, remove_prev = True):
        """download video file to location

        Args:
            movie_id (_type_): _description_
            file_location (str): file location or default
            remove_prev: remove previous file on that location

        Returns:
            True/False
        """
        result = True
        if file_location is None:
            file_location = DEFAULT_FILE_PATH
        # remove last file
        if remove_prev and os.path.exists(file_location):
            os.remove(file_location)

        url_prefix = self.url_prefix
        url_link = ''

        movie = self.movie_db.get_movie(movie_id)
        if movie:
            try:
                url_link = url_prefix + movie['url_path']
                url_link = url_link.replace(".avi", ".mp4")
                print(url_link)
                urllib.request.urlretrieve(url_link, file_location)
            except:
                print(f'An exception occurred while fetching {url_link}')
                result = False
        return result

    def save_to_db(self, movie_id, entries: List[TokenEntry]):
        error = None
        result = None
        try:
            result, error = self.movie_tokens.save_bulk_movie_tokens(movie_id, entries)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_bulk_movie_tokens: {e}'
        return result, error

    def save_to_db(self, movie_id, entry: TokenEntry):
        error = None
        result = None
        try:
            result = self.movie_tokens.save_movie_token(movie_id, entry)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_bulk_movie_tokens: {e}'
        return result, error