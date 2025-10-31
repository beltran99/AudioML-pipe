import logging

def get_logger(level=logging.DEBUG):
    return logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=level,
    )