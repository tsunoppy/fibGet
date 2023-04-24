#! /Users/tsuno/.pyenv/shims/python3
# -*- coding: utf-8 -*-
# Sectional Analysis by Fiber Model
# Coded by tsunoppy on Sunday

import math
import openpyxl
from openpyxl.utils import get_column_letter # 列幅の指定 2020/05/27

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter

import aijRc  # propeties of the steel bar material
import prop   # non-linear propeties
import pandas as pd
import store
import sqlite3


# File Control
import os

class Fiber:

    ########################################################################
    # Init
    def __init__(self,xx1,xx2,yy1,yy2,mate1,mate2):

        # addtional
        self.xx1 = xx1
        self.xx2 = xx2
        self.yy1 = yy1
        self.yy2 = yy2
        # Control Position
        # 圧縮縁のための位置データ
        self.xpos = []
        self.ypos = []
        for i in range(0,len(xx1)):
            # 左下
            self.xpos.append( xx1[i] )
            self.ypos.append( yy1[i] )
            # 右下
            self.xpos.append( xx2[i] )
            self.ypos.append( yy1[i] )
            # 左上
            self.xpos.append( xx1[i] )
            self.ypos.append( yy2[i] )
            # 右上
            self.xpos.append( xx2[i] )
            self.ypos.append( yy2[i] )

        #print( self.xpos, self.ypos )

        # material
        self.mate1 = mate1
        self.mate2 = mate2
        #print(mate1,mate2)
        self.prop_obj = []

        for i in range(0,len(mate1)):

            if mate1[i] == 1 :
                self.prop_obj.append( prop.Conc( mate2[i]) )
            elif mate1[i] == 2 :
                self.prop_obj.append( prop.St(-99,mate2[i]) )
            elif mate1[i] == 3 :
                self.prop_obj.append( prop.Conc_el( mate2[i]) )
            else:
                print("Err Prop")

        """
        # view model
        for i in range(0,len(mate1)):
            self.prop_obj[i].test()
        """

        # 収斂計算のための曲率初期値
        self.xnmax = 1.0 * 10 ** (-4)
        #self.xnmax = 1.0 * 10 ** (-5)
        #self.xnmin = 0.0
        self.xnmin = -1.0 * 10 **(-10)
        #self.xnmin = 0.5 * 10 ** (-5)

        # judge criteria
        self.eps = 1.0       # Axial Force
        self.eps2 = 10**(-6) # strain

        # concrete section
        self.x = [] # Xdir. position (m)
        self.y = [] # Ydir. position (m)
        #
        self.r = [] # Rotation, 1
        self.fc = [] # Local stiffness vector
        self.sd = [] # Local area vector
        self.ag = 0. # area
        #
        self.xg = 0. # gravity center
        self.yg = 0. # gravity center
        self.gmax = 0. # graph area
        self.gmin = 0. # graph area
        self.error = "Err." # Error Message

        # Steel Bar
        self.xs = []
        self.ys = []
        self.fy = []
        self.dia = []
        self.ra = []

        # 図芯からの位置
        self.x_xg   = []
        self.y_yg   = []
        self.xs_xg  = []
        self.ys_yg  = []


    ########################################################################
    # 収斂計算をする範囲を切り替える
    # 2022.01.28 added scripts
    def limitation(self,kappa_min,kappa_max):

        # kappa_min : minimum curvature (1/mm)
        # kappa_max : maximum curvature (1/mm)

        if kappa_min == -99 and kappa_max == -99:
            self.xnmax = 1.0 * 10 ** (-4)
            self.xnmin = -1.0 * 10 **(-10)
        elif kappa_min == -99 and kappa_max != -99:
            self.xnmax = kappa_max
        elif kappa_min != -99 and kappa_max != -99:
            self.xnmin = kappa_min
            self.xnmax = kappa_max
        elif kappa_min != -99 and kappa_max == -99:
            self.xnmin = kappa_min
        else:
            print("Err. fiber.limitation!!! " )

    ########################################################################
    # 圧縮縁を求め、回転軸から相対座標を作成
    # -- Input
    # th : 回転軸の角度
    # -- Output
    # self.dtc_th: コンクリートの相対座標
    # self.dts_th: 鉄筋の相対座標
    def rotation(self,idr,th):
        #idr: 0/Compressive fiber
        #idr: 1/Tensile fiber
        #idr: 2/Compressive Bar
        #idr: 3/Tensile Bar

        # コンクリート形状の回転軸周りの座標の入力
        dt = - math.sin(th) * np.array(self.xpos) \
            + math.cos(th) * np.array(self.ypos)
        # コンクリートファイバーの座標
        dtc = - math.sin(th) * np.array(self.x) \
            + math.cos(th) * np.array(self.y)
        # 鉄筋の座標
        dts = - math.sin(th) * np.array(self.xs) \
            + math.cos(th) * np.array(self.ys)

        # 基準となる歪
        if idr == 0:
            yyc = np.max(dt)
            #yyt = np.min(dt)
            #print("Compressive fiber :",yyc,\
            #      "Tensile fiber :",yyt)
            comment = "Compressive Concrete fiber :"
        elif idr == 1:
            yyc = np.min(dt)
            comment = "Tensile Concrete fiber :"
        elif idr == 2:
            yyc = np.max(dts)
            comment = "Compressive Steel Bar :"
        elif idr == 3:
            yyc = np.min(dts)
            comment = "Tensile Steel Bar :"
        else:
            comment = "Err strain point in self.rotation(th)"

        print(comment,yyc)
        # 相対座標の作成
        self.dtc_th = yyc - dtc
        self.dts_th = yyc - dts

    ########################################################################
    # Make Model
    def getModel(self,xx1,xx2,yy1,yy2,ndimx,ndimy,fc,\
                 ids,nx,ny,dtx,dty,dia,fy):
        # data.xlsx からデータを読み込む
        # 戻り値 0: 失敗, 1: 成功
        try:
            # xx1,xx2,yy1,yy2,kb,ndimx,ndimy
            # m,m,m,m,kN/m2/m,-,-
            for i in range(0,len(xx1)):
                self.creatMatrix(xx1[i],xx2[i],yy1[i],yy2[i],fc[i],ndimx[i],ndimy[i])

            for i in range(0,len(xx1)):
                for j in range(0,len(nx)):
                    if i == ids[j]:
                        self.createMatrix_steel(xx1[i],xx2[i],yy1[i],yy2[i],\
                                                nx[j],ny[j],dtx[j],dty[j],dia[j],fy[j])

            return 1
        except Exception as err:
            print(err)
            return 0

    ########################################################################
    # View Model
    def viewModel(self,r_model,ax,screen,outfile):

        self.viewModelGui(r_model,ax,screen,outfile)
        screen.draw()

        fig = plt.figure(figsize=(7.5/2.54,7.5/2.54))
        ax = plt.axes()
        plt.tight_layout()
        self.viewModelRep(r_model*100.0,ax,screen,outfile)
        fig.savefig(outfile+"model.png", dpi=300)
        print("save fig to",outfile+"model.png")

    ########################################################################
    def viewModelRep(self,r_model,ax,screen,outfile):

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize="8")
        ax.set_aspect('equal', 'datalim')
        # Concrete

        #ax.scatter(self.xx1,self.yy1,s=r_model,color="black")
        # Concrete
        fib = []
        for i in range(0,len(self.xx1)):
            w = self.xx2[i]-self.xx1[i]
            h = self.yy2[i]-self.yy1[i]
            fib = patches.Rectangle(xy=(self.xx1[i], self.yy1[i]), width=w, height=h, \
                                       linewidth="0.5", ec='#000000', color="gray", alpha=0.5 )
            ax.add_patch(fib)

        # Steel Bar
        c = []
        for i in range(0,len(self.xs)):
            c.append(\
                     patches.Circle(\
                                    xy=(self.xs[i],self.ys[i]),\
                                    #radius=self.dia[i]/2.0, ec='r', fill=False ))
                                    radius=self.dia[i]/2.0, fc='r' ))
        for i in range(0,len(self.xs)):
            ax.add_patch(c[i])
#                                    radius=self.dia[i]/2.0, fc='r', ec='r'))
        # Gravity Center
        sg = r_model
        ax.scatter(self.xg,self.yg, s=sg, color="blue", marker="D")

    ########################################################################
    def viewModelGui(self,r_model,ax,screen,outfile):

        # Fiber Position View
        xmax = max(self.x)
        xmin = min(self.x)
        ymax = max(self.y)
        ymin = min(self.y)
        self.gmax = max(xmax,ymax)
        self.gmin = min(xmin,ymin)

        #fig = plt.figure(figsize=(4,4))
        #ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize="8")
        ax.set_aspect('equal', 'datalim')
        #ax.set_aspect(1.0/ax.get_data_ratio(),adjustable='box')

        # Concrete
        #plot.scatter(self.x,self.y,s=r_model,color="black")
        ax.scatter(self.x,self.y,s=r_model,color="black")

        # Steel Bar
        c = []
        for i in range(0,len(self.xs)):
            c.append(\
                     patches.Circle(\
                                    xy=(self.xs[i],self.ys[i]),\
                                    #radius=self.dia[i]/2.0, fc='g', ec='r'))
                                    radius=self.dia[i]/2.0, fc='r'))
        for i in range(0,len(self.xs)):
            ax.add_patch(c[i])

        # Gravity Center
        sg = r_model*100.0
        ax.scatter(self.xg,self.yg, s=sg, color="blue", marker="D")

        """
        screen.draw()

        fig = plt.figure(figsize=(4,4))
        ax = plt.axes()
        plt.tight_layout()
        self.viewModel(r_model,ax,fig,outfile)
        fig.savefig(outfile+"model.png")
        print("save fig to",outfile+"model.png")
        """

        #plt.show()
        #plt.close(fig)

    ########################################################################
    # Make Model Matrix
    def createMatrix_steel(self,xx1,xx2,yy1,yy2,nx,ny,dtx,dty,dia,fy):

        delx = ( (xx2-xx1) - 2.0 * dtx ) / (nx-1.0)
        dely = ( (yy2-yy1) - 2.0 * dty ) / (ny-1.0)

        for i in range(0,nx):
            self.xs.append( xx1 + dtx + i * delx )
            self.ys.append( yy1 + dty )
            self.dia.append( aijRc.Aij_rc_set().Dia(dia) )
            self.fy.append(fy)
            self.ra.append( aijRc.Aij_rc_set().Ra(dia) )

        for i in range(0,nx):
            self.xs.append( xx1 + dtx + i * delx )
            self.ys.append( yy2 - dty )
            self.dia.append( aijRc.Aij_rc_set().Dia(dia) )
            self.fy.append(fy)
            self.ra.append( aijRc.Aij_rc_set().Ra(dia) )

        for i in range(0,ny-2):
            self.xs.append( xx1 + dtx )
            self.ys.append( yy1 + dty + (i+1) * dely )
            self.dia.append( aijRc.Aij_rc_set().Dia(dia) )
            self.fy.append(fy)
            self.ra.append( aijRc.Aij_rc_set().Ra(dia) )

        for i in range(0,ny-2):
            self.xs.append( xx2 - dtx )
            self.ys.append( yy1 + dty + (i+1) * dely )
            self.dia.append( aijRc.Aij_rc_set().Dia(dia) )
            self.fy.append(fy)
            self.ra.append( aijRc.Aij_rc_set().Ra(dia) )

        print("Created Steel Bar Prop.")
        fy = []
        for i in range(0,len(self.fy)):
            fy.append(self.mate2[self.fy[i]])
        self.nut = -np.dot( np.array(fy) , np.array(self.ra) )
        print("Nut=",self.nut,)
        self.etmin = self.nut/np.sum(self.ra)/(2.05*10**5)
        print(self.etmin)

    ########################################################################
    # Make Model Matrix
    def creatMatrix(self,xx1,xx2,yy1,yy2,fc,ndimx,ndimy):
        delx = (xx2-xx1)/float(ndimx)
        dely = (yy2-yy1)/float(ndimy)
        xx1_b = xx1 + delx/2.0
        yy1_b = yy1 + dely/2.0
        # print(delx,dely)
        # Create spring position
        for j in range(0,ndimy):
            for i in range(0, ndimx):
                self.x.append(float(xx1_b+float(i)*delx))
                self.y.append(float(yy1_b+float(j)*dely))
                self.r.append(float(1))
        """
        # Check Data read
        k=1
        for j in range(0,ndimy+1):
        for i in range(0, ndimx+1):
        idx = i + (ndimx+1) * j
        #print(i,j,"ID=",idx,"Position[m]=",x[idx],y[idx])
        k = k+1
        print("xsize=", len(x), "ysize=",len(y))
        """
        # Create spring properties
        # local area
        da = delx*dely
        for j in range(0,ndimy):
            for i in range(0,ndimx):
                # For Corner
                self.fc.append(fc)
                self.sd.append(da)
                # print(i,j,kd[i+j*ndimx])

        print("Created Cocncrete Fiber Prop.")

    ########################################################################
    # Make Gravity center
    def getG(self,xx1,xx2,yy1,yy2):
        tmpag = 0.
        tmpxg = 0.
        tmpyg = 0.

        suma = 0.
        sumxg = 0.
        sumyg = 0.

        for i in range(0,len(xx1)):
            tmpag = (xx2[i]-xx1[i])*(yy2[i]-yy1[i])
            tmpxg = (xx2[i]+xx1[i])/2.0
            tmpyg = (yy2[i]+yy1[i])/2.0
            suma = suma + tmpag
            sumxg = sumxg + tmpxg*tmpag
            sumyg = sumyg + tmpyg*tmpag

        self.xg = float(sumxg)/float(suma)
        self.yg = float(sumyg)/float(suma)
        self.ag = float(suma)


        # xg,yg からの相対座標系を作成
        for i in range(0,len(self.x)):
            self.x_xg.append( self.x[i] - self.xg )
            self.y_yg.append( self.y[i] - self.yg )
        for i in range(0,len(self.xs)):
            self.xs_xg.append( self.xs[i] - self.xg )
            self.ys_yg.append( self.ys[i] - self.yg )

        # Save Model Input
        savefile = "./db/input.txt"
        lines = "## Center of Gravity\n"
        lines += " "
        lines += "gx = {:.2f} mm".format(self.xg)+", "
        lines += "gy = {:.2f} mm".format(self.yg)+", "
        lines += "A = {:.2e} mm2".format(self.ag)+"\n"
        #self.out_add(savefile,lines)
        self.out(savefile,lines)

        print(lines)

    ########################################################################
    # write output text file
    def out(self,outFile,lines):
        fout = open(outFile, "w")
        fout.writelines(lines)
        fout.close()

    # add output text file
    def out_add(self,outFile,lines):
        fout = open(outFile, "a")
        fout.writelines(lines)
        fout.close()


    ########################################################################
    # 指定された歪から軸力を算定する。
    def nn0(self,e):

        nnc = 0.0
        for i in range(0, len(self.x) ):
            #nnc = nnc + prop.Conc(self.fc[i]).sig_c(e) * self.sd[i]
            nnc = nnc + self.prop_obj[self.fc[i]].sig_c(e) * self.sd[i]

#            print ( i, prop.Conc(self.fc[i]).sig_c(e) )

        nns = 0.0

        sigmax = -10000.0
        sigmin =  10000.0

        for i in range(0, len(self.xs) ):
            #sig = prop.St(-99,self.fy[i]).sig_s(e)
            sig = self.prop_obj[self.fy[i]].sig_s(e)
            nns = nns +  sig * self.ra[i]

            # max or min
            if sig > sigmax: sigmax = sig
            if sig < sigmin: sigmin = sig
#            print( i, self.fy[i], self.ra[i], prop.St(-99,self.fy[i]).sig_s(e) )

        nnc = nnc/1000.0
        nns = nns/1000.0
        #print(nns,nnc)

        return ( nns + nnc ), sigmax, sigmin


    ########################################################################
    # 指定された軸力から圧縮歪を算定する。
    # use self.nn0
    def e0(self,nn):

        #eps = 10 ** (-3) # judge of the itelation
        #e1 =  0.001*10**(-2)
        e1 = self.etmin
        e2 =  0.3*10**(-2)

        kk1 = self.nn0(e1)[0] - nn
        kk2 = self.nn0(e2)[0] - nn

        for i in range(0,10000):

            e = 0.5 * ( e1 + e2 )

            kk1 = self.nn0(e1)[0] - nn
            kk2 = self.nn0(e2)[0] - nn

            """
            if abs(kk2) < self.eps2: break;

            e0 = e2
            e2 = ( kk2*e1 - kk1 * e2 ) / ( kk2 - kk1 )
#            if abs(e2 - e1) > abs(e2-e0):
#                e1 = e0
            e1 = e0
            #print(i,kk2+nn)
            """

            kk = self.nn0(e)[0] - nn
            if abs(kk) < self.eps: break;
            if kk * kk1 > 0 :
                e1 = e
            else:
                e2 = e


        #e0 = e2
        e0 = e
        tmp, sigmax, sigmin = self.nn0(e0)
        print(sigmax,sigmin)

        print("e0 by N, Count=",i, "EPS=", abs(kk2), "e0=", e0,e1)

        return e0, sigmax,sigmin

    ########################################################################
    # 圧縮縁の歪みと中立軸からコンター図を作成
    def view_sig_c(self,ec0,xn,th,ax,screen):

        # Stress plot
        #fig = plt.figure(figsize=(4,4))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize="8")
        ax.set_aspect('equal')

        es_c = ec0 - xn * self.dtc_th

        sig = []
        for i in range(0, len(self.x) ):
            #sig.append( prop.Conc(self.fc[i]).sig_c(es_c[i]) )
            sig.append( self.prop_obj[self.fc[i]].sig_c(es_c[i]) )
            #print(self.fc[i])

        cont = ax.scatter(self.x,self.y,c=sig,cmap='jet',marker=",")
        #ax.colorbar(cont)

        screen.draw()
        #plt.show()
        #plt.close(fig)


    ########################################################################
    # 圧縮縁の歪みと中立軸から軸力を算定
    def nnc_e(self,ec0,xn,th):

        # ID:0, ec0:圧縮歪み
        # ID:1, ec0:軸歪み

        # For Concrete
        #dt_c = math.sin(th) * np.array(self.x) + math.cos(th) * np.array(self.y)
        #es_c = ec0 * ( xn ** np.array(self.r) - dt_c ) / xn
        #es_c = ec0 * ( xn - dt_c ) / xn
        ## this is
        #es_c = ec0 - xn * dt_c

        es_c = ec0 - xn * self.dtc_th

        nnc = 0.0
        for i in range(0, len(self.x) ):
            #print(prop.Conc(self.fc[i]).sig_c(es_c[i]))
            nnc = nnc + self.prop_obj[self.fc[i]].sig_c(es_c[i]) * self.sd[i]
            #nnc = nnc + prop.Conc[self.fc[i]].sig_c(es_c[i]) * self.sd[i]

        nnc = nnc/1000.0

        # For Steel Bar
        #dt_s = math.sin(th) * np.array(self.xs) + math.cos(th) * np.array(self.ys)

        ## this is
        es_s = ec0 - xn * self.dts_th

        nns = 0.0
        for i in range(0, len(self.xs) ):
            #sig = prop.St(-99,self.fy[i]).sig_s(es_s[i])
            sig = self.prop_obj[self.fy[i]].sig_s(es_s[i])
            nns = nns +  sig * self.ra[i]

        nns = nns/1000.0

        #print(es_c,nnc,es_s,nns)

        return nnc+nns

    ########################################################################
    # 軸力と曲げモーメントから圧縮歪みを算定
    # under devolop
    def mn(self,mm,nn,th,ee1,ee2):

        e1 = ee1
        e2 = ee2

        #e1 = 0.1/10**2
        #e2 = 0.3/10**2

        # addtional
        aa, bb, cc, eemax, eemin, eesmax, eesmin, ecmin\
            = self.mm_ec_xn(e1,self.xnn(e1,nn,th,-99),th)
        kk1 = bb - mm
        aa, bb, cc, eemax, eemin, eesmax, eesmin, ecmin\
            = self.mm_ec_xn(e2,self.xnn(e2,nn,th,-99),th)
        kk2 = bb - nn
        ##

        for i in range(0,10000):

            """
            # addtional
            e = 0.5 * ( e1 + e2 )
            ##
            """

            aa, bb, cc, eemax, eemin, eesmax, eesmin, ecmin\
                = self.mm_ec_xn(e1,self.xnn(e1,nn,th,-99),th)
            kk1 = bb - mm
            aa, bb, cc, eemax, eemin, eesmax, eesmin, ecmin\
                = self.mm_ec_xn(e2,self.xnn(e2,nn,th,-99),th)
            kk2 = bb - nn

            """
            # addtional
            aa, bb, cc, eemax, eemin, eesmax, eesmin\
                = self.mm_ec_xn(e2,self.xnn(e,nn,th,-99),th)
            kk = bb - nn

            if abs(kk) < self.eps: break;
            if kk * kk1 > 0 :
                e1 = e
            else:
                e2 = e
            ##
            """


            if abs(kk2) < self.eps: break;

            e0 = e2
            e2 = ( kk2*e1 - kk1 * e2 ) / ( kk2 - kk1 )
            # ????
            e2 = abs(e2)

            #print(ec0,xn0,xn1,xn2,kk1,kk2)
#            if abs(xn2 - xn1) > abs(xn2-xn0):
#                xn1 = xn0
            e1 = e0


        e0 = e2

        print("-- MN Count=",i,\
              "mm={:10.6e}".format(mm),"EPS={:10.6e}".format(abs(kk2)),\
              "ec= {:10.6e}".format(e0),\
              "N= {:10.0f}".format(kk2+nn) )

        return e0

    ########################################################################
    # 圧縮縁の歪みと中立軸から曲げモーメントと曲率を算定
    def mm_ec_xn(self,ec0,xn,th):

        # For Concrete
        #dt_c = math.sin(th) * np.array(self.x) + math.cos(th) * np.array(self.y)
        #es_c = ec0 * ( xn ** np.array(self.r) - dt_c ) / xn
        #es_c = ec0 * ( xn - dt_c ) / xn
        # this is
        #es_c = ec0 - xn * dt_c
        es_c = ec0 - xn * self.dtc_th

        mmcx = 0.0
        mmcy = 0.0

        nnc = 0.0

        for i in range(0, len(self.x) ):
            #sig = prop.Conc(self.fc[i]).sig_c(es_c[i])
            sig = self.prop_obj[self.fc[i]].sig_c(es_c[i])
            mmcx = mmcx + sig * self.sd[i] * self.x_xg[i]
            mmcy = mmcy + sig * self.sd[i] * self.y_yg[i]

        mmcx = mmcx/10**(6)
        mmcy = mmcy/10**(6)

        # For Steel Bar
        #dt_s = math.sin(th) * np.array(self.xs) + math.cos(th) * np.array(self.ys)

        ## this is
        es_s = ec0 - xn * self.dts_th

        mmsx = 0.0
        mmsy = 0.0

        sigmax = -10000.0
        sigmin =  10000.0

        for i in range(0, len(self.xs) ):
            #sig = prop.St(-99,self.fy[i]).sig_s(es_s[i])
            sig = self.prop_obj[self.fy[i]].sig_s(es_s[i])

            mmsx = mmsx +  sig * self.ra[i] * self.xs_xg[i]
            mmsy = mmsy +  sig * self.ra[i] * self.ys_yg[i]
            if sig > sigmax: sigmax = sig
            if sig < sigmin: sigmin = sig

        mmsx = mmsx/10**(6)
        mmsy = mmsy/10**(6)

        #print(es_c,nnc,es_s,nns)

        return xn, (mmcx + mmsx), (mmcy + mmsy), sigmax, sigmin, np.max(es_s), np.min(es_s), np.min(es_c)

    ########################################################################
    # 圧縮歪みと軸力から曲率を逆算
    def xnn(self,ec0,nn,th,ctl):
        # xn1,xn2: initial value

        #eps = 10 ** (-3) # judge of the itelation

        if ctl == -99 :
            """
            xn1 = self.xnmin
            xn2 = self.xnmax
            """
            # addtional
            xn1 = self.xnmin
            xn2 = self.xnmax
            #####

        elif ctl == -98:

            # addtional
            xn1 = self.xnmin
            xn2 = self.xnmax/10.0
            #####


        else:
            xn1 = 0.5*ctl
            xn2 = 1.5*ctl

            #xn1 = self.xnmin
            xn2 = self.xnmax

        # addtional
        kk1 = self.nnc_e(ec0,xn1,th) - nn
        kk2 = self.nnc_e(ec0,xn2,th) - nn
        # this is
        #xn1 = 1.0 * 10 ** (-12)
        #xn1 = 0.5 * 10 ** (-10)
        #xn2 = 1.0 * 10 ** (-10)


        #print("ec0",ec0,"ctl",ctl)

        for i in range(0,10000):

            # add
            xn = 0.5 * ( xn1 + xn2 )

            kk1 = self.nnc_e(ec0,xn1,th) - nn
            kk2 = self.nnc_e(ec0,xn2,th) - nn

            # add
            kk = self.nnc_e(ec0,xn,th) - nn
            if abs(kk) < self.eps: break;
            if kk * kk1 > 0 :
                xn1 = xn
            else:
                xn2 = xn

            # Check
            #print(i,kk1,kk2,kk,xn1,xn2)

        xn0 = xn

        """

            if abs(kk2) < self.eps: break;

            #print(xn1,xn2,kk2,kk1,kk2-kk1)
            xn0 = xn2
            xn2 = ( kk2*xn1 - kk1 * xn2 ) / ( kk2 - kk1 )
            # ????
            #xn2 = abs(xn2)
#            print(ec0,xn0,xn1,xn2,kk1,kk2)
#            if abs(xn2 - xn1) > abs(xn2-xn0):
#                xn1 = xn0
            xn1 = xn0

        xn0 = xn2
        """

        print("-- Count=",i,\
              "ec={:10.6e}".format(ec0),"EPS={:10.6e}".format(abs(kk)),\
              "κ= {:10.6e}".format(xn0),\
              "N= {:10.0f}".format(kk+nn) )

        return xn0

    ########################################################################
    # 圧縮縁の歪みと軸力から中立軸を求める。
    #    def xn_nn(self,ec0,th,nn):


    ########################################################################
    # make conter
    # コンクリートの圧縮歪からコンター図を作成
    def make_cont(self,nn,th,eu,ax,screen):

        # nn: axial force
        # theta: angle
        # eu: strain at extream compressive fiber
        # ax,screen: data for plot
                # Ultimate Strength
        ########################################
        th = th/360.0 * 2.0 * math.pi
        self.rotation(0,th)
        ctl = -99
        pu , mux, muy, eemax, eemin, eesmax, eesmin, ecmin\
            = self.mm_ec_xn(eu,self.xnn(eu,nn,th,ctl),th)

        # make concter
        ########################################
        self.view_sig_c(eu,self.xnn(eu,nn,th,ctl),th,ax,screen)

        print("# Result, at ultimate strain")
        print("########")
        print("eu  = {:15.6e}".format(eu), "-")
        print("pu  = {:15.6e}".format(pu), "1/mm")
        print("mux = {:15.0f}".format(mux), "kN.m")
        print("muy = {:15.0f}".format(muy), "kN.m")



    # 指定した歪みに対する解析
    # return capacity corresponding to the specified strain
    ########################################################################
    def solveBySt(self,nn,theta,idr,e,title):

        # Ultimate Strength
        ########################################
        #idr: 0/Compressive fiber
        #idr: 1/Tensile fiber
        #idr: 2/Compressive Bar
        #idr: 3/Tensile Bar


        th = theta/360.0 * 2.0 * math.pi
        self.rotation(idr,th)

        if idr == 1:
            ctl = -98
            e0, sigmax, sigmin = self.e0(nn)
        else:
            ctl = -99

        if idr == 1 and e0 <= e:
            pu = 0.0
            mux = 0.0
            muy = 0.0
            eemax = 0.0
            eemin = 0.0
            eesmax = 0.0
            eesmin = 0.0
            ecmin = 0.0

        else:
            pu , mux, muy, eemax, eemin, eesmax, eesmin, ecmin\
                = self.mm_ec_xn(e,self.xnn(e,nn,th,ctl),th)

        comment = "--------------------\n"
        comment += title
        comment += "\n"
        comment += "ec  = {:15.6e} -\n".format(e)
        comment += "es  = {:15.6e} -\n".format(eesmin)
        comment += "φ   = {:15.6e} 1/mm\n".format(pu)
        comment += "mux = {:15.0f} kN.m\n".format(mux)
        comment += "muy = {:15.0f} kN.m\n".format(muy)

        print(comment)

        """
        # addtional parameter for the steel bar strained, if specified
        self.parameter = pu
        """

        #return comment,e,eesmin,pu,mux,muy,ecmin
        return comment, \
            pu , mux, muy, eemax, eemin, eesmax, eesmin, ecmin\


    ########################################################################
    # mn generator
    # under dev
    def mnGen(self,ndiv,theta,nn,ecu,esu,esa,ax,screen):

        #ax.clear()

        #fig2 = plt.figure()
        #ax  = plt.axes()

        nd = []
        numin = self.nut/10**3
        numax = self.nn0(ecu)[0]

        nd.append(self.nut/10**3)
        mx = [0.0]
        my = [0.0]

        deln = (numax-numin)/(ndiv+1)

        for i in range(0,ndiv):

            nn = numin + deln * float(i+1)
            comment, pu , mux, muy, eemax_u, eemin_u, eesmax_u, eesmin_u, ecmin_u =\
                self.solveBySt(nn,theta,0, ecu,"# Ultimate by Concrete")


            mx.append(mux)
            my.append(muy)
            nd.append(nn)

        mx.append(0.0)
        my.append(0.0)
        nd.append(numax)


        # plot
        ####################
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(np.abs(mx),nd,label='Mx',c='black')
        ax.plot(np.abs(my),nd,label='My',c='black',linestyle='dotted')

        print("MN data")
        for i in range(0,len(nd)):
            print(nd[i],mx[i],my[i])

        #ax.scatter(mx,my)

        #ax.axhline(y=0,color='black',linewidth=0.5,linestyle='--')
        #ax.axvline(x=0,color='black',linewidth=0.5,linestyle='--')
        ax.legend(fontsize=8)

        ax.set_xlim(0,)

        ax.set_xlabel("Bending Moment [kN.m]",fontsize=8)
        ax.set_ylabel("Axial Force [kN]",fontsize=8)
        ax.tick_params(labelsize="8")
        ax.grid()
        ax.set_aspect('auto')

        screen.draw()
        #plt.show()


    ########################################################################
    def mxmy_double(self,ndiv,nn,ecu,esu,esa,ax,screen):

        mx = []
        my = []

        mx2 = []
        my2 = []

        delth = 360.0 / ndiv

        for i in range(0,ndiv):

            theta = i * delth

            print("angle = ",theta,"!!!")

            comment, pu , mux, muy, eemax_u, eemin_u, eesmax_u, eesmin_u, ecmin_u =\
                self.solveBySt(nn,theta,0, ecu,"# Ultimate by Concrete")

            if eesmin_u < -esu:
            #print("eesmin_u",eesmin_u,"esu",esu,"bad condition")
                self.limitation(-99,pu)
                comment, pu , mux, muy, eemax_u, eemin_u, eesmax_u, eesmin_u, ecmin_u =\
                    self.solveBySt(nn,theta,3,-esu,"# Ultimate by Steel Bar")
                self.limitation(-99,-99)

            mx.append(mux)
            my.append(muy)

            if eesmin_u < -esa:
                self.limitation(-99,pu)
                comment, pa , ma_x, ma_y, eemax_a, eemin_a, eesmax_a, eesmin_a, ecmin_a =\
                    self.solveBySt(nn,theta,3, -esa,"# Allowable by Steel Bar")
                self.limitation(-99,-99)
            else:
                ma_x = mux
                ma_y = muy

            mx2.append(ma_x)
            my2.append(ma_y)


        mx.append(mx[0])
        my.append(my[0])
        mx2.append(mx2[0])
        my2.append(my2[0])


        xmax = max( abs(np.max(mx)), abs(np.min(mx)) )
        ymax = max( abs(np.max(my)), abs(np.min(my)) )
        xymax = max ( xmax, ymax ) * 1.2

        # plot
        ####################
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(mx,my,label='ultimate',c='black')
        ax.plot(mx2,my2,label='steel bar at allowable',c='black',linestyle='dotted')
        #ax.scatter(mx,my)

        ax.axhline(y=0,color='black',linewidth=0.5,linestyle='--')
        ax.axvline(x=0,color='black',linewidth=0.5,linestyle='--')
        ax.legend(fontsize=8)
        ax.grid()

        ax.set_xlim(-xymax,xymax)
        ax.set_ylim(-xymax,xymax)

        ax.set_xlabel("Mx [kN.m]",fontsize=8)
        ax.set_ylabel("My [kN.m]",fontsize=8)
        ax.tick_params(labelsize="8")
        ax.set_aspect('equal')

        ax.grid()

        screen.draw()

        return mx,my


    """
    ########################################################################
    # 軸力一定時において、thを動かして、Mx-My関係を求める。
    # update, 2022/02/04 , to be no used

    def mxmy(self,nn,idr,e,ndiv,ax,screen):

        mx = []
        my = []

        delth = 2.0*math.pi / ndiv

        for i in range(0,ndiv):

            th = i * delth
            self.rotation(idr,th)
            ctl = -99

            pu , mux, muy, eemax, eemin, eesmax, eesmin, ecmin\
                = self.mm_ec_xn(e,self.xnn(e,nn,th,ctl),th)

            mx.append(mux)
            my.append(muy)

        mx.append(mx[0])
        my.append(my[0])

        xmax = max( abs(np.max(mx)), abs(np.min(mx)) )
        ymax = max( abs(np.max(my)), abs(np.min(my)) )
        xymax = max ( xmax, ymax ) * 1.2

        # plot
        ####################
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(mx,my,label='ultimate')
        ax.plot(mx2,my2,label='steel bar at allowable')
        #ax.scatter(mx,my)

        ax.axhline(y=0,color='black',linewidth=0.5,linestyle='--')
        ax.axvline(x=0,color='black',linewidth=0.5,linestyle='--')
        ax.legend(fontsize=8)
        ax.grid()

        ax.set_xlim(-xymax,xymax)
        ax.set_ylim(-xymax,xymax)

        ax.set_xlabel("Mx [kN.m]",fontsize=8)
        ax.set_ylabel("My [kN.m]",fontsize=8)
        ax.tick_params(labelsize="8")
        plt.ax.set_aspect('equal')

        ax.grid()

        screen.draw()

        return mx,my
    """

    ########################################################################
    # Solve Fiber analysis
    # 軸力一定にて、M-φ関係を求める。
    def solve(self,nn,theta,ecumax,ndiv,eu,esu,ax,screen,id_cal,dbname,outfile):
        # nn : axial force (kN)
        # theta : angle for the strain to global cordinate
        # eumax : maximum strain of compression

        details = "### Analysis detail ----------- \n"
        details += "# Start Solve------\n"
        print(details)

        # Preparation Cal

        th = theta/360.0 * 2.0 * math.pi
        print("theta=",theta,"sin=",math.sin(th),"cos=",math.cos(th))
        print("ecumax=",ecumax,"ndiv=",ndiv)
        self.rotation(0,th)

        # initial conctrol
        e0, sig0max, sig0min = self.e0(nn)
        if e0 > 0 :
            del_e = ( ecumax - e0 ) / (ndiv-1)
        elif e0 <= 0 :
            del_e = ecumax / (ndiv-1)

        del_e = ( ecumax - e0 ) / (ndiv-1)

        phai = []
        mx   = []
        my   = []
        mm   = []

        phai2 = []
        emax = []
        emin = []
        esmax = []
        esmin = []

        phai.append(0.0)
        mx.append(0.0)
        my.append(0.0)
        mm.append(0.0)

        phai2.append(0.0)
        emax.append(sig0max)
        emin.append(sig0min)
        esmax.append(e0)
        esmin.append(e0)

        # 圧縮歪み
        ctl_ec = []
        ctl_ec.append(0.0)

        # Calculation
        for i in range(0,ndiv):

            # 圧縮歪み

            if e0 > 0.0:
                e = e0 + float(i) * del_e
            else:
                e = float(i+1) * del_e

            e = e0 + float(i) * del_e

            ctl_ec.append(e)

            # 中立軸を求める
            #self.xnn(e,nn,th)
            if i == 0 :
                ctl = -99
#                aa, bb, cc, eemax, eemin, eesmax, eesmin\
#                    = self.mm_ec_xn(eu,self.xnn(eu,nn,th,ctl),th)
#                print(aa)
#                ctl = aa
            aa, bb, cc, eemax, eemin, eesmax, eesmin, ecmin\
                = self.mm_ec_xn(e,self.xnn(e,nn,th,ctl),th)
            ctl = aa
            phai.append(aa)
            mx.append(bb)
            my.append(cc)
            mm.append( math.sqrt(bb**2 + cc**2) )

            phai2.append( aa )
            emax.append( eemax )
            emin.append( eemin )
            esmax.append( eesmax )
            esmin.append( eesmin )

        print("# End Cal ----- ")

        ctl = -99

        # Allowable Strength
        ########################################
        #ea = prop.Conc(fc[0]).ecs(fc[0]*2.0/3.0)
        ea = self.prop_obj[self.fc[0]].ecs(self.mate2[self.fc[0]]*2.0/3.0)
        print(ea)
        pa , mxa, mya, eemax, eemin, eesmax, eesmin, ecmin\
            = self.mm_ec_xn(ea,self.xnn(ea,nn,th,ctl),th)
        print("# Result, at allowable strain")
        print("########")
        print("ea  = {:15.6e}".format(ea), "-")
        print("pa  = {:15.6e}".format(pa), "1/mm")
        print("mxa = {:15.0f}".format(mxa), "kN.m")
        print("mya = {:15.0f}".format(mya), "kN.m")

        # Ultimate Strength
        ########################################
        pu , mux, muy, eemax, eemin, eesmax, eesmin, ecmin\
            = self.mm_ec_xn(eu,self.xnn(eu,nn,th,ctl),th)

        print("# Result, at ultimate strain")
        print("########")
        print("eu  = {:15.6e}".format(eu), "-")
        print("pu  = {:15.6e}".format(pu), "1/mm")
        print("mux = {:15.0f}".format(mux), "kN.m")
        print("muy = {:15.0f}".format(muy), "kN.m")


        # data save
        # Save Model Input
        ########################################################################
        savefile = "./db/tmp.csv"
        lines = "p,mx,my,emax,emin,esmax,esmin,ec,xn\n"

        xn = []
        xn.append(0.0)

        for i in range(0,len(mx)):
            if i >= 1:
                xn.append(ctl_ec[i]/phai[i])
            lines += str(phai[i]) + "," + str(mx[i]) + "," +  str(my[i]) + \
                "," + str(emax[i]) + "," + str(emin[i]) + \
                "," + str(esmax[i]) + "," + str(esmin[i]) + \
                "," + str(ctl_ec[i]) + "," + str(xn[i]) + "\n"

        #self.out_add(savefile,lines)
        self.out(savefile,lines)
        self.out(outfile+"mp",lines)

        # making data to sqlite3
        df = pd.read_csv(savefile)
        #print(df,len(df))
        #print(df)
        table = str(id_cal) + 'mp'
        obj = store.Store(dbname)
        obj.make_table(savefile,table)

        """
        #
        # plot graph
        ########################################################################
        esmax = np.array(esmax) *100
        esmin = np.array(esmin) *100
        # make M-phai relationship, steel bar stress, strain
        ########################################
        self.view_mp("|M|","Mx","My",pa,mxa,mya,pu,mux,muy,ax[0],screen[0],table,dbname)
        self.view_steel_stress(phai2,emax,emin,"stress, N/mm2","Max. Stress","Min. Stress",ax[1],screen[0])
        self.view_steel_stress(phai2,esmax,esmin,"strain, %","Max. Strain","Min. Strain",ax[2],screen[0])
        """


########################################################################
# End Class

########################################################################

class AftFib:

    # example
    # data plot by sqlite data base
    ########################################################################
    # fiber.AftFib('test.db').plotGui(table,ax,screen)

    ########################################################################
    # initial data
    def __init__(self,dbname,id_draw):
        # dbname: data of sqlite
        #         or
        #         cntl csvfile
        self.dbname = dbname
        # index of plot function
        # id_draw == 0 : plot to gui
        # id_draw == 1 : save figure
        #id_draw = 0
        self.id_draw = id_draw

    ########################################################################
    # Plot
    def plotGui(self,id_cal,ax,screen):

        ####################
        # read data

        df2 = pd.read_csv(self.dbname)
        cuvmax     = df2.iloc[id_cal,9]
        mumax      = df2.iloc[id_cal,10]
        stressmax  = df2.iloc[id_cal,11]
        strainmax  = df2.iloc[id_cal,12]

        pathname = os.path.dirname(self.dbname)
        outfile = pathname + "/" + df2.iloc[id_cal,13].replace(' ','')
        df = pd.read_csv(outfile+"mp")

        if strainmax != -99:
            strainmax = strainmax*100.0

        # reading data
        p = np.array(df["p"])
        mx = np.array(df["mx"])
        my = np.array(df["my"])
        mxmy = np.sqrt( mx**2 + my**2 )
        emax = np.array(df["emax"])
        emin = np.array(df["emin"])
        esmax = np.array(df["esmax"])*100.0
        esmin = np.array(df["esmin"])*100.0


        # for capacity
        df2 = pd.read_csv(outfile+"cap")
        p_cap = np.array(df2["p"])
        mx_cap = np.array(df2["mx"])
        my_cap = np.array(df2["my"])

        print(df2)

        # figure.add_subplot(111)

        if self.id_draw == 0:

            self.view_mp(p,mxmy,mx,my,"|M|","Mx","My",cuvmax,mumax,ax[0],screen[0],\
                         p_cap,mx_cap,my_cap)

            self.view_steel_stress(\
                                   p,emax,emin,\
                                   "stress, N/mm2","Max. Stress","Min. Stress",\
                                   cuvmax,stressmax,ax[1],screen[0])

            self.view_steel_stress(p,esmax,esmin,\
                                   "strain, %","Max. Strain","Min. Strain",\
                                   cuvmax,strainmax,ax[2],screen[0])
        else:
            # ax = self.matplotlib_axes***
            # screen = self.matplotlib_canvas2
            #        = figure.add_subplot(111)
            """
            figure2 = self.matplotlib_figure2 = Figure(tight_layout=True)
            self.matplotlib_axes2 = figure2.add_subplot(4,1,(1,2))
            self.matplotlib_axes3 = figure2.add_subplot(413)
            self.matplotlib_axes4 = figure2.add_subplot(414)
            self.matplotlib_canvas2 = FigureCanvas(self, wx.ID_ANY, figure2)
            """
            # ax = plt.axes(), screen = plt.figure()
            screen = plt.figure(figsize=(9/2.54,12/2.54)) # same as fig
            ax0 = screen.add_subplot(4,1,(1,2))
            ax1 = screen.add_subplot(413)
            ax2 = screen.add_subplot(414)
            self.view_mp(p,mxmy,mx,my,"|M|","Mx","My",cuvmax,mumax,ax0,screen,p_cap,mx_cap,my_cap)
            self.view_steel_stress(p,emax,emin,"stress, N/mm2","Max. Stress","Min. Stress",cuvmax,stressmax,ax1,screen)
            self.view_steel_stress(p,esmax,esmin,"strain, %","Max. Strain","Min. Strain",cuvmax,strainmax,ax2,screen)
            #screen.savefig(outfile+"mp.png", dpi=300)
            screen.savefig(outfile+"mp.png",dpi=300)
            print("save fig to",outfile+"mp.png")

    """
    ########################################################################
    # Plot by sqlite3
    def plotGui2(self,id_cal,ax,screen):

        ####################
        # read data
        #dbname = './db/test.db'
        #table = '0mp'
        table = str(id_cal)+'mp'
        conn = sqlite3.connect(self.dbname)
        df  = pd.read_sql_query('SELECT * FROM "%s"' % table, conn)
        df2 = pd.read_sql_query('SELECT * FROM "CNTL"', conn)

        cuvmax     = df2.iloc[id_cal,9]
        mumax      = df2.iloc[id_cal,10]
        stressmax  = df2.iloc[id_cal,11]
        strainmax  = df2.iloc[id_cal,12]
        if strainmax != -99:
            strainmax = strainmax*100.0

        conn.close()
        #print(df)
        #print(df["p"])

        # data making
        p = np.array(df["p"])
        mx = np.array(df["mx"])
        my = np.array(df["my"])
        mxmy = np.sqrt( mx**2 + my**2 )
        emax = np.array(df["emax"])
        emin = np.array(df["emin"])
        esmax = np.array(df["esmax"])*100.0
        esmin = np.array(df["esmin"])*100.0

        if self.id_draw == 0: # plot to gui
            self.view_mp(p,mxmy,mx,my,"|M|","Mx","My",cuvmax,mumax,ax[0],screen[0])
            self.view_steel_stress(p,emax,emin,"stress, N/mm2","Max. Stress","Min. Stress",cuvmax,stressmax,ax[1],screen[0])
            self.view_steel_stress(p,esmax,esmin,"strain, %","Max. Strain","Min. Strain",cuvmax,strainmax,ax[2],screen[0])
        else:
            # ax = self.matplotlib_axes***
            # screen = self.matplotlib_canvas2
            #        = figure.add_subplot(111)
            # ax = plt.axes(), screen = plt.figure()
            screen = plt.figure() # same as fig
            ax0 = screen.add_subplot(4,1,(1,2))
            ax1 = screen.add_subplot(413)
            ax2 = screen.add_subplot(414)
            self.view_mp(p,mxmy,mx,my,"|M|","Mx","My",cuvmax,mumax,ax0,screen)
            self.view_steel_stress(p,emax,emin,"stress, N/mm2","Max. Stress","Min. Stress",cuvmax,stressmax,ax1,screen)
            self.view_steel_stress(p,esmax,esmin,"strain, %","Max. Strain","Min. Strain",cuvmax,strainmax,ax2,screen)
            screen.show
            print("under dev")
    """

    ########################################################################
    # M-p relationship
    def view_mp(self,p,mxmy,mx,my,mmlabel,mxlabel,mylabel,xmax,ymax,\
                ax,screen,\
                p_cap,mx_cap,my_cap):
        # mmlabel,mxlabe,mylabel : label
        # xmax,ymax: maxmimum axis value ( if == -99 --> auto scale output )
        # ax,screen: ax = plt.axes(), screen = plt.figure()
        if ymax > 100000:
            ax.yaxis.offsetText.set_fontsize(8)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y",scilimits=(0,0))



        #screen.rcParams["font.size"]=8
        # make graph
        ax.plot(p,np.abs(np.array(mx)),label=mxlabel,marker=".",c='black',)
        ax.plot(p,np.abs(np.array(my)),label=mylabel,marker=".",c='black',markerfacecolor='None',linewidth = 0.5)
        ax.scatter(p_cap,np.abs(np.array(mx_cap)),label="capacity,mx",color="r",marker="s")
        ax.scatter(p_cap,np.abs(np.array(my_cap)),label="capacity,my",color="b",marker="D")
        #ax.plot(p,mxmy,label="M",marker="None",c='red')
        ax.legend(fontsize=8)
        #,pa,mxa,mya,pu,mux,muy
        """
        # 短期
        ax.plot(pa,abs(mxa),marker="s", c="red")
        ax.plot(pa,abs(mya),marker="s", c="red")
        #終局
        ax.plot(pu,abs(mux),marker="s", c="red")
        ax.plot(pu,abs(muy),marker="s", c="red")
        """

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("Curvature [1/mm]",fontsize=8)
        #ax.set_ylabel("Curvature [1/mm]",fontsize=8)
        #ax.set_title("Bending Moment [kN.m]",loc='left',fontsize=8)
        ax.set_title("Bending Moment [kN.m]",loc='center',fontsize=8)
        ax.xaxis.offsetText.set_fontsize(8)
        ax.tick_params(labelsize="8")
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x",scilimits=(0,0))
        ax.grid()
        if xmax == -99:
            ax.set_xlim(0,)
        else:
            ax.set_xlim(0,xmax)
        if ymax == -99:
            ax.set_ylim(0,)
        else:
            ax.set_ylim(0,ymax)

        #ax.hlines([0],"black")
        #plt.show()
        #plt.close(fig)
        if self.id_draw == 0:
            screen.draw()
        else:
            #plt.show()
            screen.tight_layout()
            #screen.show()
            #print("under dev mp??")

    ########################################################################
    # Steel bar stress
    def view_steel_stress(self,phai2,emax,emin,title,emaxlabel,eminlabel,\
                          xxmax, yymax,ax,screen):

        ax.plot(phai2,emax,label=emaxlabel,marker=".",c='black',markerfacecolor='None',linestyle="dotted")
        ax.plot(phai2,emin,label=eminlabel,marker=".",c='black')
        ax.legend(fontsize=8)
        ax.grid()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize="8")
        ax.set_xlabel("Curvature [rad/mm]",fontsize=8)
        ax.set_title(title,loc='left',fontsize=8)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(8)

        if xxmax == -99:
            ax.set_xlim(0,)
        else:
            ax.set_xlim(0,xxmax)

        if yymax != -99:
            ax.set_ylim(-yymax,yymax)


        if self.id_draw == 0:
            screen.draw()
        #else:
            #plt.show()
            #screen.tight_layout()
            #screen.show()
            #print("under dev steel??")


if __name__ == "__main__":

    readFile = os.path.join(os.path.dirname(__file__), 'input3.csv')
    theta,ecumax,ndiv,nn,ecu,esu,\
        mate1,mate2,\
        xx1,yy1,xx2,yy2,ndimx,ndimy,fc,\
        ids,nx,ny,dtx,dty,dia,fy,\
        =\
        read_data(readFile) # need modified

    print("------------------------------")
    obj = Fiber(xx1,xx2,yy1,yy2,mate1,mate2)

    if obj.getModel(xx1,xx2,yy1,yy2,ndimx,ndimy,fc,\
                    ids,nx,ny,dtx,dty,dia,fy):
        obj.getG(xx1,xx2,yy1,yy2)
        obj.viewModel(0.5)
        print("Complete Model Making")
    else:
        del obj
        obj = Fiber()
        print("Fail Model Making")

    obj.solve(nn,theta,ecumax,ndiv)
    #print( "N  =", obj.nn0(0.2*10**(-3)) )
    #print( "e0 =", obj.e0(9832.0) )
    #obj.nn0(-5.0*10**(-4))
    print("------------------------------")
    #obj.solve(nn,0,0)
    print("Complete")


