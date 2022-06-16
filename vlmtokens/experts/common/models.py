from dataclasses import dataclass
from numpy import array
from .constants import OUTPUT_JSON

@dataclass
class ExpertParam:
    movie_id: str
    scene_element: int = None
    local: bool = False
    extra_params: dict = None
    output: str = OUTPUT_JSON

@dataclass
class TokenRecord:
    movie_id: str
    scene_element: int = 0
    scene: int = 0
    expert: str = None
    bbox: list = None
    label: str = None
    meta_label: dict = None
    re_id: int = 0



