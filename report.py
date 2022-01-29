#! /Users/tsuno/.pyenv/shims/python3
# -*- coding:utf-8 -*-
import os, sys
#import Image
#import urllib2
#from cStringIO import StringIO


#zipアーカイブからファイルを読み込むため。通常は必要ないはず。
#sys.path.insert(0, 'reportlab.zip')

import reportlab
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm

########################################################################
import pandas as pd
import sqlite3

#
import linecache
#

class Report():

    def __init__(self,cntlfile):
        self.cntlFile = cntlfile
        self.pathname = os.path.dirname(self.cntlFile)
        #self.FONT_NAME = "Helvetica"
        self.FONT_NAME = "GenShinGothic"
        GEN_SHIN_GOTHIC_MEDIUM_TTF = "./fonts/GenShinGothic-Monospace-Medium.ttf"
        # フォント登録
        pdfmetrics.registerFont(TTFont('GenShinGothic', GEN_SHIN_GOTHIC_MEDIUM_TTF))
        #font_size = 20
        #c.setFont('GenShinGothic', font_size)

    ########################################################################
    # 文字と画像を配置
    def create_row(self,c, index, df, df2, index2):
        #y_shift = -240 * index
        y_shift = -360 * index
        #y_shift = -180 * index
        c.setFont(self.FONT_NAME, 9)
        """
        for i in range(0,len(data)):
            # txt
            c.drawString(300, 720-(i-1)*10 + y_shift, data[i])
        """
        #c.drawString(55, self.ypos(0,y_shift), data[0].encode('utf-8'))
        #c.drawString(55, self.ypos(1,y_shift), data[1].encode('utf-8'))

        # Slab Condition
        #lx = "{:.2f}".format(float(data[2]))
        #ly = "{:.2f}".format(float(data[3]))
        mc_x = "{:10.0f}".format(df.iloc[0,1])
        mc_y = "{:10.0f}".format(df.iloc[0,2])
        ma_x = "{:10.0f}".format(df.iloc[1,1])
        ma_y = "{:10.0f}".format(df.iloc[1,2])
        mu_x = "{:10.0f}".format(df.iloc[2,1])
        mu_y = "{:10.0f}".format(df.iloc[2,2])


        title = str(df2[0])
        theta = str(df2[2])
        nn    = str(df2[3])
        comment = df2[8].replace(' ','')

        # Model
        c.setFont(self.FONT_NAME, 12)
        c.drawString(55, self.ypos(1,y_shift),
                     title\
                     )
        c.setFont(self.FONT_NAME, 9)
        c.drawString(60, self.ypos(3,y_shift),
                     comment,\
                     )
        c.drawString(65, self.ypos(4,y_shift),
                     "N=" + nn + "kN,  "\
                     "θ=" + theta + "deg."\
                     )


        # Design Condition
        c.setFont(self.FONT_NAME, 9)

        c.drawString(55, self.ypos(27,y_shift),
                     "Capacity:"\
                     )
        c.drawString(60, self.ypos(29,y_shift),
                     "Mcx = " + mc_x + " kN.m"\
                     )
        c.drawString(60, self.ypos(30,y_shift),
                     "Max = " + ma_x + " kN.m"\
                     )
        c.drawString(60, self.ypos(31,y_shift),
                     "Mux = " + mu_x + " kN.m"\
                     )

        c.drawString(180, self.ypos(29,y_shift),
                     "Mcy = " + mc_y + " kN.m"\
                     )
        c.drawString(180, self.ypos(30,y_shift),
                     "May = " + ma_y + " kN.m"\
                     )
        c.drawString(180, self.ypos(31,y_shift),
                     "Muy = " + mu_y + " kN.m"\
                     )

        # png
        #imagefile="./db/"+str(index2+index)+"mp.png"
        imagefile=self.pathname + "/" + df2.iloc[13].replace(' ','') + "mp.png"
        #print("Index=",index2+index)
        #c.drawImage(imagefile, 250,  y_shift + 470, width=5*cm , preserveAspectRatio=True)
        #c.drawImage(imagefile, 300,  y_shift + 280, width=9*cm , preserveAspectRatio=True)
        c.drawImage(imagefile, 300,  y_shift - 130, width=9*cm, preserveAspectRatio=True, mask='auto')

        #print(os.listdir('./db'))
        #os.remove(imagefile)
        #print(os.listdir('./db'))

        # model
        #imagefile="./db/"+str(index2+index)+"model.png"
        imagefile=self.pathname + "/" + df2.iloc[13].replace(' ','') + "model.png"
        #c.drawImage(imagefile, 50,  y_shift + 420, width=8*cm , preserveAspectRatio=True)
        c.drawImage(imagefile, 50,  y_shift + 150, width=7.5*cm, preserveAspectRatio=True, mask='auto')

    def ypos(self,ipos,y_shift):
        return 730-(ipos-1)*10 + y_shift

    ########################################################################
    # pdfの作成
    def print_page(self, c, index, nCase):


        #タイトル描画
        c.setFont(self.FONT_NAME, 20)
        #c.drawString(50, 795, u"Design of the twoway slab")
        c.drawString(50, 795, u"Fiber Analysis")

        #グリッドヘッダー設定
        #xlist = [40, 380, 560]
        xlist = [40, (40+560)/2, 560]
        ylist = [760, 780]
        c.grid(xlist, ylist)

        #sub title
        c.setFont(self.FONT_NAME, 12)
        c.drawString(55, 765, u"Model")
        c.drawString(315, 765, u"M-φ Relationship")

        #データを描画
        ########################################################################
        #for i, data in range(0,int(nCase)):

        #dbname = './db/test.db'
        #conn = sqlite3.connect(dbname)

        for i in range(0,nCase):


            """
            table = 'CNTL'
            df2 = pd.read_sql_query('SELECT * FROM "%s"' % table, conn)
            df2 = df2.iloc[i+index,:]
            """
            df2 = pd.read_csv(self.cntlFile)
            df2 = df2.iloc[i+index,:]

            """
            table = str(index+i)+'cap'
            df = pd.read_sql_query('SELECT * FROM "%s"' % table, conn)
            data = df
            """
            table = self.pathname + "/" + df2.iloc[13].replace(' ','') + "cap"
            df  = pd.read_csv(table)
            data = df

            #print(df2)

            #line = linecache.getline('./db/rcslab.txt', index+i+1 )
            #data = line.split(', ')

            #linecache.clearcache()
            #f.close()
            #data = tmpData
            self.create_row( c, i, data, df2, index )

        #conn.close()
        #最後にグリッドを更新
        #ylist = [40,  280,  520,  760]
        ylist = [40,  400,  760]
        #ylist = [40,  160, 280, 400, 520, 640, 760]
        #ylist = [40,  220,  400, 580, 760]4bunnkatsu
        c.grid(xlist, ylist[2 - nCase:])
        #ページを確定
        c.showPage()

    ########################################################################
    # pdfの作成
    def print_head(self, c , title):

        #title = 'Sample Project'

        #タイトル描画
        c.setFont(self.FONT_NAME, 20)
        c.drawString(50, 795, title)

        #sub title
        c.setFont(self.FONT_NAME, 12)

        #データを描画
        ########################################################################
        inputf = './db/input.txt'
        f = open(inputf,'r', encoding='utf-8')
        tmpData = []
        while True:
            line = f.readline()
            if line:
                if line != '\n':
                    tmpData.append(line.replace('\n',''))
                else:
                    tmpData.append('')
            else:
                break
        f.close()
        data = tmpData
        #c.setFont(self.FONT_NAME, 9)
        for i in range(0,len(data)):
            # txt
            c.drawString(55, 720-(i-1)*14, data[i])
        """
        # Model Diagram
        imagefile = './db/model.png'
        c.drawImage(imagefile, 60,  -300, width=18*cm , preserveAspectRatio=True)
        """
        #ページを確定
        c.showPage()
    ########################################################################
    # whole control
    def create_pdf(self, dataNum, pdfFile, title):

        # Parameter -------
        # inputf   : path to text file
        # imagefile: path to png file
        # pdfFile  : name of making pdf file

        #フォントファイルを指定して、フォントを登録
        #folder = os.path.dirname(reportlab.__file__) + os.sep + 'fonts'
        #pdfmetrics.registerFont(TTFont(FONT_NAME, os.path.join(folder, 'ipag.ttf')))
        #出力するPDFファイル
        c = canvas.Canvas(pdfFile)

        # ページ数
        ########################################################################
        #dataNum = len(inputf)
        numPage = dataNum // 2
        numMod = dataNum % 2
        #print(numPage,numMod)
        if numMod >= 1:
            numPage = numPage + 1

        # pdfの作成
        ########################################################################
        #self.print_head( c , title)

        for i in range(0,numPage):
            index = 2*i # index: 参照データのインデックス
            if numPage == 1:
                self.print_page( c, index, dataNum)
            elif i != numPage-1 and numPage != 1:
                self.print_page( c, index, 2)
            else:
                if numMod != 0:
                    self.print_page( c, index, numMod)
                else:
                    self.print_page( c, index, 2 )

        #pdfファイル生成
        ########################################################################
        c.save()
        print ("repot.py is Okay!!.")

########################################################################
# END CLASS


"""
########################################################################
# test script

pathname = "./test.pdf"
obj = Report()
# テキストの読み込み
########################################################################
inputf = []
inputf.append("./db/rcslab.txt")
inputf.append("./db/rcslab.txt")
inputf.append("./db/rcslab.txt")
inputf.append("./db/rcslab.txt")
inputf.append("./db/rcslab.txt")
inputf.append("./db/rcslab.txt")

title = "sample"

obj.create_pdf(3,pathname,title)
"""

#title = 'sample'
#pathname = "./test.pdf"
#Report().create_pdf(15,pathname,title)
