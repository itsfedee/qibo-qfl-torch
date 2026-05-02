import warnings
import logging
import os
import numpy as np

def setup_logging():
    # Python warnings
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Ray environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['RAY_DEDUP_LOGS'] = '0'