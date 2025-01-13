import os
import json
import logging
import functools
from datetime import datetime
from functionals.system_config import ModelConfig




def log_to_file(func):
    file_path = os.path.abspath(__file__)
    file_path = os.path.dirname(os.path.dirname(file_path))
    file_path = os.path.join(file_path, 'logs')
    now = datetime.now().strftime("%Y%m%d_%H")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 获取函数名称作为日志文件名的基础
        log_filename = f"{file_path}/{func.__name__}_{now}.log"

        # 配置日志
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger(func.__name__)
        # 开始记录日志
        logger.info(f"函数 '{func.__name__}' 开始运行.")

        # 执行函数
        result = func(*args, **kwargs)
        logger.info(result)

        return result

    return wrapper


def debug(json_data, json_name):
    config = ModelConfig()
    now = datetime.now().strftime("%Y%m%d")
    with open(f"{config.file_path}/logs/debug/{json_name}_{now}.josn", "w") as f:
        json.dump(json_data, f, indent=4)


# 示例使用
if __name__ == "__main__":
    @log_to_file
    def example_function():
        print("This is some output.")
        return "Function executed successfully."

    # 调用被装饰的函数
    result = example_function()
    print(f"Result: {result}")
