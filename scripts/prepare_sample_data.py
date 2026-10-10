"""Copy missing sample inputs into the local outputs directory."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import OUTPUT_DIR, initialize_sample_data


if __name__ == '__main__':
    initialize_sample_data()
    print(f'示例数据已准备：{OUTPUT_DIR}')
