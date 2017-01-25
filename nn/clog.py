import logging


clog = logging.getLogger() # 'root' Logger
console = logging.StreamHandler()
format_str = '[%(levelname)s]\t%(asctime)s > %(filename)s:%(lineno)s | %(message)s'
console.setFormatter(logging.Formatter(format_str))
clog.addHandler(console)
clog.setLevel(logging.INFO)
