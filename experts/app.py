import os
import sys
from queue import Queue
import logging
import json
from threading import Thread
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from .service.base_expert import BaseExpert
from nebula3_experts.nebula3_pipeline.pipeline.api import PipelineApi
import experts.common.constants as constants
from experts.common.defines import ExpertCommands, OutputStyle
from experts.common.models import ExpertParam
from experts.common.config import ExpertConf


tags_metadata = [
    {
        "name": "status",
        "description": "View running status of pipeline steps.",
    },
    {
        "name": "set",
        "description": "Set a configuration: <cfg_name>=<value> where cfg_name is one of the configurations \
        run 'cfg' command to see possible configurations.",
    },
    {
        "name": "cfg",
        "description": "list all editable configurations.",
    },
    {
        "name": "tasks",
        "description": "list all running tasks.",
    },
    {
        "name": "predict",
        "description": "run the expert on the specific movie and location. with the given output",
    },
]

class PredictParam(BaseModel):
    movie_id: str
    scene_element: int = None
    local: bool
    extra_params: dict = None
    output: str = constants.OUTPUT_JSON

    class Config:
        schema_extra = {
            "example": {
                "movie_id": "the movie id in db",
                "scene_element": "movie's scene element",
                "local": "movie location: local (true) /remote (false)",
                "extra_params": "the expert's specific params in json object",
                "output": "where to output: json (return json in response)/db, default- db"
            }
        }


class ExpertApp:
    def __init__(self, expert: BaseExpert, params = None):
        self.params = params
        self.expert = expert
        self.running = True
        self.logger = self.init_logger()
        self.msgq = Queue()
        self.config = ExpertConf()
        # check env for disable pipeline
        if self.config.get_run_pipeline():
            self.pipeline =  PipelineApi(self.logger) #PIPELINE_API(self.logger)
            self.init_pipeline()
        else:
            self.pipeline = None
        self.app = FastAPI(openapi_tags=tags_metadata)
        self.add_base_apis()
        self.expert.set_logger(self.logger)
        # Add expert specific apis
        self.expert.add_expert_apis(self.app)

    def __del__(self):
        if not self.running:
            self.running = False
            self.msgq.put_nowait(constants.STOP_MSG)
            if (self.msg_thread and self.msg_thread.is_alive()):
                self.msg_thread.join()

    def init_logger(self):
        """
        configure the global logger:
        - write DEBUG+ level to stdout
        - write ERROR+ level to stderr
        - format: [time][thread name][log level]: message
        @param log_file: the file to which we wish to write. if an existing dir is given, log to a file
                        labeled with the curent date and time. if None, use the current working directory.
        """
        logger_name = f'{self.expert.get_name()}-{os.getpid()}'
        # create a logger for this instance
        logger = logging.getLogger(logger_name)

        # set general logging level to debug
        logger.setLevel(logging.DEBUG)

        # choose logging format
        formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s]: %(message)s')

        # create stdout stream handler
        shdebug = logging.StreamHandler(sys.stdout)
        shdebug.setLevel(logging.DEBUG)
        shdebug.setFormatter(formatter)
        logger.addHandler(shdebug)

        # create stderr stream handler
        sherr = logging.StreamHandler(sys.stderr)
        sherr.setLevel(logging.ERROR)
        sherr.setFormatter(formatter)
        logger.addHandler(sherr)

        title = f'===== {logger_name} ====='
        logger.info('=' * len(title))
        logger.info(title)
        logger.info('=' * len(title))

        return logger

    def init_pipeline(self):
        """init pipeline params
        and add pipeline event handlers/callbacks
        """
        self.pipeline.subscribe(self.expert.get_name(),
                                self.expert.get_dependency(),
                                self.on_pipeline_msg)
        self.msg_thread = Thread(target=self.msg_handle)
        # TODO enable this for pipeline
        # self.msg_thread.start()

    def on_pipeline_msg(self, msg):
        """handle msg received from pipeline subscription
        Args:
            msg (_type_): msg received
        """

        # log msg put to queue
        self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.PIPELINE_NOTIFY, constants.PARAMS: msg })

    def msg_handle(self):
        """handle incoming msg
        """
        msg_iterator = iter(self.msgq, constants.STOP_MSG)
        while self.running:
            try:
                msg = next(msg_iterator)
            except StopIteration:
                # log exit
                if self.expert.handle_exit:
                    self.expert.handle_exit()
                self.running = True
            else:
                # log msg received
                cmd = msg[constants.COMMAND]
                params = msg[constants.PARAMS]
                if ExpertCommands.PIPELINE_NOTIFY == cmd:
                    self.logger.debug(f'Recieved PIPELINE_NOTIFY: {params[constants.MOVIE_ID]}')
                    self.expert.handle_msg(params)
                elif ExpertCommands.CLI_PREDICT == cmd:
                    self.logger.debug(f'Recieved CLI_PREDICT: {params[constants.MOVIE_ID]}')
                    self.expert.handle_msg(params)
                else:
                    msg_json = json.dumps(msg)
                    self.logger.error(f'Received unknown: {msg_json}')


    def add_base_apis(self):
        """add base apis
        Returns:
            _type_: _description_
        """
        @self.app.get("/")
        def read_root():
            return {"Expert": self.expert.get_name()}


        @self.app.get("/status")
        def get_status():
            return {"status": self.expert.get_status()}

        @self.app.post("/set", tags=['set'] )
        def post_config():
            return {"set": 'not supported' }

        @self.app.get("/cfg", tags=['cfg'] )
        def get_config():
            return {"cfg": self.expert.get_cfg()}

        @self.app.get("/tasks", tags=['tasks'] )
        def get_tasks():
            return {"tasks": self.expert.get_tasks() }

        @self.app.post("/predict", tags=['run'] )
        async def predict(params: PredictParam):
            """ predict - this is an async command so putting to queue """
            # parsing into ExpertParams
            expert_params = self.parse_params(params)
            if expert_params.output == constants.OUTPUT_DB:
                self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.CLI_PREDICT, constants.PARAMS: expert_params })
            # no output style is like json -> predict and return result
            return self.expert.predict(expert_params)

    def parse_params(self, params):
        expert_params = ExpertParam(params.movie_id,
                                    params.scene_element,
                                    params.local,
                                    params.extra_params,
                                    params.output)
        return expert_params

    def run(self):
        print("Running...")
        # self.expert.run()

    def get_app(self):
        return self.app

