#! /Users/tsuno/.pyenv/shims/python3
# -*- coding: utf-8 -*-

# Calculation for Steel stiffnering following Building letter By BCJ
# Coded by tsunoppy on Sunday

import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

########################################################################
# Concrete Elastic Prop.
class Conc_el:

    def __init__(self,sig_b):

        self.es = 4700.0*math.sqrt(sig_b)
        self.ea = sig_b / self.es

    def sig_c(self,e):

        if e >= 0.0:
            return self.es * e
        else:
            return self.es * e / 1000.0

    ########################################################################
    # コンクリートが短期許容応力度に達する時の歪み
    def ecs(self,sig_s):
        return sig_s/self.es

    ########################################################################
    # コンクリートの履歴曲線
    def test(self):

        x = np.arange( -2.0*self.ea, 2.0*self.ea, 2.0*self.ea/100.0 )
        y = []
        #print (x)
        for para in x:
            #print(para,self.sig_c(para))
            y.append( self.sig_c(para) )

        y = np.array(y)
        plt.plot(x, y, 'b-')
        plt.grid()
        plt.show()


########################################################################
# Concrete Nonlinear Prop.
class Conc:

    ########################################################################
    def __init__(self, sig_b ):

        # N/mm2 to kg/cm2 : 10.1972
        # kg/cm2 to N/mm2 :  0.0980665
        self.tokg = 10.1972
        self.tosi = 0.0980665

        ##
        # Dimension
        self.sig_b = sig_b  # compressive concrete strength (N/mm2)
        self.ft = - 1.8* math.sqrt(self.sig_b*self.tokg)*self.tosi  # Tension Strength (N/mm2), negative

        # Prop.
        # Yang Modulus (N/mm2)
        self.ec    = 4.0 * ( self.sig_b * self.tokg / 1000.0 )**(0.333) * 10 **5
        self.ec    = self.ec * self.tosi
        # strain at ultimate
        self.e0    = 0.5243 * (self.sig_b *self.tokg ) ** (0.25) * 10.0 ** (-3)

        # strain at tension strength
        self.et    = self.ft/self.ec
        self.eu    = 10.0 * self.et


        # con for Fafitis-Shah
        self.alpha = self.ec/ self.sig_b * self.e0
        self.c = 1.67 * self.sig_b * self.tokg

        """
        print("# Make Object")
        print("## Inp.")
        print("   sigb = {:.2f} N/mm2".format(self.sig_b))
        print(" ## Prop.")
        print("   Ec   = {:.0f} N/mm2".format(self.ec))
        print(" ## Compression.")
        print("   α    = {:.5f} ".format(self.alpha))
        print("   c    = {:.5f} kg/cm2".format(self.c))
        print("   e0   = {:.5f} ".format(self.e0))

        print(" ## Tension.")
        print("   ft   = {:.2f} N/mm2".format(self.ft))
        print("   et   = {:.2e} ".format(self.et))
        print("   eu   = {:.2e} ".format(self.eu))
        """

    ########################################################################
    def sig_c(self, e):

        # e: concrete strain

        if e <= self.eu:
            return 0.0

        elif self.eu < e and e <= self.et:
            # Tension Stiffning
            xx = ( -e + self.et ) / ( -self.eu + self.et )
            return \
                self.ft * ( 1.0 - 8.0/3.0 * xx + 3.0 * xx**2 - 4.0/3.0 * xx**3 )

        elif self.et < e and e <= 0.0:
            # Tension under elastic stage
            return self.ec * e

        elif 0 < e and e <= self.e0 :
            # Compression up to ultimate
            return \
                self.sig_b * ( 1.0 - ( 1.0-e/self.e0 )**(self.alpha) )

        elif self.e0 < e:
            # Compression after ultimate
            s = self.sig_b * math.exp( - self.c * (e-self.e0)**1.15 )
            if s <= 0.0 :
                return 0.0
            return \
                self.sig_b * math.exp( - self.c * (e-self.e0)**1.15 )
        else:
            print("error sig_c!, e=",e)

        #if e < 0.0:
        #s = 0.0

        ########################################################################
        #if e < 0.0:
        #x = 0.0
        ########################################################################

        #return s

    ########################################################################
    # コンクリートが短期許容応力度に達する時の歪み
    def ecs(self,sig_s):
        return self.e0 * ( 1.0 - ( 1.0 - sig_s/self.sig_b )**(1.0/self.alpha) )

    ########################################################################
    # コンクリートが引張強度に達する時の歪み
    def ect(self):
        return self.et

    ########################################################################
    # コンクリートの履歴曲線
    def test(self):

        fig = plt.figure()
        ax = plt.axes()

        x = np.arange( self.eu*1.5, self.e0*5, (self.e0-self.eu*4)/100.0 )
        y = []
        #print (x)
        for para in x:
            #print(para,self.sig_c(para))
            y.append( self.sig_c(para) )

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("strain")
        ax.set_ylabel("stress [N/mm2]")
        plt.tight_layout()

        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)

        y = np.array(y)
        #plt.plot(x, y, 'b-')
        plt.plot(x, y, 'black')
        plt.grid()
        plt.show()

        #fig.savefig("test.png")

    # matoplot
    #https://note.nkmk.me/python-matplotlib-patches-circle-rectangle/
    ##########################
    def model(self,ax,canv):

        fig = plt.figure()
        #ax = plt.axes()

        #ax.axis('scaled')
        #ax.set_aspect('equal')
        #ax.axis("off")

        #ax.set_ylim(hh-1450, hh+50)
        #ax.set_xlim(-50, 1500)
        #plt.savefig('./db/sample.jpg')
        #plt.plot([0,0],[self.c1/2,self.c2/2]) だめ

        """
        plt.show()
        plt.close(fig)
        """
        x = np.arange( self.eu*1.5, self.e0*4, (self.e0-self.eu*4)/100.0 )
        y = []

        for para in x:
            y.append( self.sig_c(para) )

        y = np.array(y)

        ax.girid()
        ax.plot(x, y, 'b-')

        canv.draw()


    ########################################################################
    def image_pdf(self,imagefile):

        fig = plt.figure()
        ax = plt.axes()

        x = np.arange( self.eu*1.5, self.e0*4, (self.e0-self.eu*4)/100.0 )
        y = []

        for para in x:
            y.append( self.sig_c(para) )

        y = np.array(y)

        ax.plot(x, y, 'b-')

        plt.savefig(imagefile)

########################################################################
# End Class


# 鉄の履歴
class St:

    ####################
    def __init__(self, es, fy ):

        if es == -99:
            self.es = 2.05* 10**5
            #self.es = 2.05* 10**5 * 0.102
        else:
            self.es = es
        self.fy = fy

        # this is!
        #self.est = self.es*0.1
        self.est = self.es

        self.ey = self.fy/self.es
        self.ey2 = self.fy/self.est

        self.es2 = self.es / 100.0

        """
        print("# Make Object")
        print("## Inp.")
        print("   Es = {:.2f} N/mm2".format(self.es))
        print("   fy = {:.0f} N/mm2".format(self.fy))
        """

    ####################
    # cal. stress corresponding to the input strain
    def sig_s(self,e):

        if e < -self.ey2:
            return -self.fy - self.es2 * ( abs(e) - self.ey2 )

        elif -self.ey2 <= e and e <= 0.0:
            return self.est * e

        elif 0.0 < e and e <= self.ey:
            return self.es * e

        elif self.ey < e:
            return self.fy + self.es2 * ( e - self.ey )

    ####################
    # cal. strain corresponding to the input stress
    def st_s(self,sig):

        return sig/self.es

    ####################
    # 履歴曲線
    def test(self):

        x = np.arange( -self.ey*20.0, self.ey*20.0, self.ey/100.0 )
        y = []
        #print (x)
        for para in x:
            y.append( self.sig_s(para) )

        y = np.array(y)
        plt.plot(x, y, 'b-')
        plt.show()


########################################################################
# End Class


# imput data

"""
fc = 60.
concrete = Conc(60.0)
concrete.test()


steel = St(-99,490)
steel.test()
"""
#concrete.image_pdf("test.jpg")



