import sqlite3


class SQLWriter:
    def __init__(self, db_path='intent.db'):
        self.db_path = db_path  # the name of the database
        self.tb_name = 'intent'  # the name of the table

    def create_table(self):
        """Create a dataset named self.db_path, with only one table named self.tb_name"""
        # check if the file exists
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS intent (time_stamp text, args text, best_tr_f1_macro real, best_tr_acc real, best_tr_epoch int, best_va_f1_macro real, best_va_acc real, best_va_epoch int, current_epoch int, tr_time real, va_time real, terminated boolean, PRIMARY KEY (time_stamp))''',)  # time_stamp is the main key. timestamp is a reserved word and cannot be used
        conn.commit()
        conn.close()

    def write_entry(self, time_stamp: str, args: str, best_tr_f1_macro: float, best_tr_acc: float, best_tr_epoch: int, best_va_f1_macro: float, best_va_acc: float, best_va_epoch: int, current_epoch: int, tr_time: float, va_time: float, terminated: bool):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("REPLACE INTO intent VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (time_stamp, args, best_tr_f1_macro, best_tr_acc, best_tr_epoch, best_va_f1_macro, best_va_acc, best_va_epoch, current_epoch, tr_time, va_time, terminated))
        conn.commit()
        conn.close()

    def read(self, show=True, criterion='all', ordering='time'):  # ordering is either time or f1
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        answer = []
        if ordering == 'time':
            order_by = 'time_stamp'
        elif ordering == 'f1':
            order_by = 'best_va_f1_macro'
        elif ordering == 'acc':
            order_by = 'best_va_acc'
        if criterion == 'all':
            query = f'SELECT * FROM intent ORDER BY {order_by} ASC'
        elif criterion == 'terminated':
            query = f'SELECT * FROM intent WHERE terminated=1 ORDER BY {order_by} ASC'
        elif criterion == 'cnn':
            query = f'''SELECT * FROM intent WHERE args LIKE '%cnn%' ORDER BY {order_by} ASC'''  # refer to https://www.w3schools.com/sql/sql_like.asp
        for row in c.execute(query):
            if show:
                print(row)
            answer.append(row)
        conn.close()
        return answer
