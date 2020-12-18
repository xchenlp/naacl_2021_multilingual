from sql_writer import SQLWriter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--db_path', type=str, default='intent.db')
parser.add_argument('--ordering', type=str, default='time', help='time, f1, or acc')
args = parser.parse_args()


if __name__ == '__main__':
    sql_writer = SQLWriter(db_path=args.db_path)
    all_tasks = sql_writer.read(ordering=args.ordering)
