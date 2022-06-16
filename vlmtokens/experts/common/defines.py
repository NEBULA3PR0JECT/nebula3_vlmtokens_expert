import enum

class ExpertStatus(enum.Enum):
    STARTING = 1
    ACTIVE = 2
    RUNNING = 2

class ExpertCommands(enum.Enum):
    CLI_PREDICT = 1
    PIPELINE_NOTIFY = 2

class OutputStyle(enum.Enum):
    JSON = 1
    DB = 2

class ExpertTask:
    id: str
    info: dict