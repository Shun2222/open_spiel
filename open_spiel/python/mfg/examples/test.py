import logger
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/mnt/shunsuke/mfg_result", help="save dir")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])
    logger.record_tabular(f'log test1', 1)
    logger.record_tabular(f'log test2', 2)
    logger.dump_tabular()
    logger.record_tabular(f'log test3', 3)
    logger.record_tabular(f'log test4', 4)
