import os
import zipfile
from pathlib import Path
from urllib import request as req
from End_to_End_ML_Pipeline import logger
from End_to_End_ML_Pipeline.utils.common import get_size
from End_to_End_ML_Pipeline.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = req.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"File: {filename} downloaded with info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok = True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)