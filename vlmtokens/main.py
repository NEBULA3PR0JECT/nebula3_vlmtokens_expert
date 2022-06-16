# from typing import Optional
from fastapi import FastAPI

# tags_metadata = [
#     {
#         "name": "status",
#         "description": "View running status of pipeline steps.",
#     },
#     {
#         "name": "set",
#         "description": "Set a configuration: <cfg_name>=<value> where cfg_name is one of the configurations \
#         run 'cfg' command to see possible configurations.",
#     },
#     {
#         "name": "cfg",
#         "description": "list all editable configurations.",
#     },
# ]

# app = FastAPI(openapi_tags=tags_metadata)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/status")
# def get_status(q: Optional[str] = None):
#     return {"status": "running" }

# @app.post("/set}", tags=['set'] )
# def post_status(q: Optional[str] = None):
#     return {"set": 'ok', "q": q }

# @app.get("/cfg}", tags=['cfg'] )
# def get_config(q: Optional[str] = None):
#     return {"set": 'ok', "q": q }

from typing import Optional

from experts.service.base_expert import BaseExpert
from experts.app import ExpertApp
from experts.common.defines import *
from experts.common.models import ExpertParam

import sys
# sys.path.insert(0, 'nebula3_vlm/')

class MyExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        # after init all
        self.set_active()

    def get_name(self):
        return "MyExpert"

    def add_expert_apis(self, app: FastAPI):
        @app.get("/my-expert")
        def get_my_expert(q: Optional[str] = None):
            return {"expert": "my-expert" }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.movie_id)
        print(f'Predicting movie: {expert_params.movie_id}')
        return { 'result': { 'movie_id' : expert_params.movie_id, 'info': movie , 'extra_params': expert_params.extra_params} }

my_expert = MyExpert()
expert_app = ExpertApp(expert=my_expert)
app = expert_app.get_app()
expert_app.run()
