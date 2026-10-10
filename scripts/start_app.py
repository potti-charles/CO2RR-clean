"""Launch the app from any directory; load local .env without replacing the environment."""

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env_file = root / '.env'
    if env_file.exists():
        for number, raw_line in enumerate(env_file.read_text(encoding='utf-8').splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            name, separator, value = line.partition('=')
            name = name.strip()
            if not separator or not name.isidentifier():
                raise ValueError(f'.env 第 {number} 行格式错误，应为 KEY=value')
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(name, value)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv[0] = str(root / 'app.py')
    runpy.run_path(str(root / 'app.py'), run_name='__main__')


if __name__ == '__main__':
    main()
