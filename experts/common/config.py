import os

class ExpertConf:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExpertConf, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.RUN_PIPELINE = eval(os.getenv('EXPERT_RUN_PIPELINE', 'False'))

    def get_run_pipeline(self):
        return (self.RUN_PIPELINE)
