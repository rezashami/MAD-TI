from pathlib import Path
from datetime import datetime
import logging

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self):

        rootPath = ".\\all logs\\"
        Path(rootPath).mkdir(parents=True, exist_ok=True)
        outputFileName = rootPath + datetime.now().strftime("%Y-%m-%d %H-%M-%S")+'.log'
        logging.basicConfig(filename=outputFileName,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filemode='w')

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def log(self, level, msg):
        if level == "debug":
            self.logger.debug(msg)
        elif level == "info":
            self.logger.info(msg)
        elif level == "error":
            self.logger.error(msg)
