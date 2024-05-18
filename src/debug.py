import sys
from enum import Enum, unique

from yaml import safe_load

from llamafactory.eval.evaluator import run_eval
from llamafactory.train.tuner import export_model, run_exp
from llamafactory.webui.interface import run_web_demo, run_web_ui


@unique
class Command(str, Enum):
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    HELP = "help"


def main():
    command = sys.argv.pop(1)
    if command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        args = None
        with open("./config/llama3_config.yaml", "r", encoding="utf-8") as f:
            args = safe_load(f)
        run_exp(args)
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    else:
        raise NotImplementedError("Unknown command: {}".format(command))


main()
