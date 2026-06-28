import logging
import os

os.makedirs("logs",exist_ok=True)

logger = logging.getLogger('Akki')
logger.setLevel('DEBUG')

# getting file handler
file_handler = logging.FileHandler('logs/error.log')
file_handler.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


# formatter 
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)




