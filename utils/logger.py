import logging


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    # 根据层级__name__获取logger,确保每次获取到同一个实例
    logger = logging.getLogger(__name__)
    # 设置进行日志的级别,只有级别高于level的信息才会被记录
    # 后续通过logger.debug(msg) logger.info(msg)等方式记录日志信息
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    # 设置日志记录的格式formatter
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    # 流处理器:负责处理日志信息
    # console = logging.StreamHandler()
    # console.setLevel(log_level if rank == 0 else 'ERROR')
    # console.setFormatter(formatter)
    # 给logger注册标准输入输出流处理器
    # logger.addHandler(console)
    # 给logger注册文件handler,用来持久化保存日志信息
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # 关闭日志传递 a.b 的日志不会传给 a
        logger.propagate = False
    return logger
    