import os.path as osp
import warnings
warnings.filterwarnings('ignore')
from predict_st.api import BaseExperiment
from predict_st.utils import (create_parse, default_parse, get_dist_info, load_condig, update_config)

if __name__ == '__main__':
    args = create_parse().parse_args()
