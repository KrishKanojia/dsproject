from src.dsproject.logger import logging
from src.dsproject.exception import CustomException
import sys

if __name__ == "__main__":
    logging.info("Hello World!")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)