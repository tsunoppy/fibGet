#! /Users/tsuno/.pyenv/shims/python3
import sqlite3
import pandas as pd

class Store:

    ########################################################################
    # init data
    def __init__(self,dbname):
        self.dbname     = dbname

    ########################################################################
    # make database
    def make_table(self,input_file,tablename):

        # read csv file by pandas
        ####################
        df = pd.read_csv(input_file)
        ####################
        conn = sqlite3.connect(self.dbname)
        #cur = conn.cursor()
        df.to_sql(tablename,conn,if_exists='replace',index=None)
        conn.close()

    ########################################################################
    # make database
    def conv_pd_data(self,df,tablename):

        conn = sqlite3.connect(self.dbname)
        df.to_sql(tablename,conn,if_exists='replace',index=None)
        conn.close()

    ########################################################################
    # read database
    def conv_csv(self,tablename,out_file):
        conn = sqlite3.connect(self.dbname)
        df=pd.read_sql_query('SELECT * FROM %s' % tablename, conn)
        print(df)
        df.to_csv(out_file,header=True,index=None)
        conn.close()


########################################################################
# end of class
"""
dbname = 'TEST.db'
input_file = './tmp.csv'
tablename = 'mp'
out_file = 'tmp000.csv'

st = Store(dbname)
st.make_table(input_file,tablename)
st.conv_csv(tablename,out_file)

print(os.path.dirname(self.showFileDialog()))
"""
