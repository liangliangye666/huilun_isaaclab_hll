"""`python -m l5a_sim2sim` 的入口。

实际的命令行解析和运行编排都放在 `cli.main()`，这里保持极薄入口，方便
同时支持 `python -m l5a_sim2sim` 和 pyproject 中注册的 `l5a-sim2sim` 命令。
"""

from .cli import main


if __name__ == "__main__":
    main()
