#! /Users/tsuno/.pyenv/shims/python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan  6 16:07:40 2021
@author: Tsunoppy
"""
class Aij_rc_set():

    #
    # Young Modulus
    def Ec(self,fc,gamma):
        # gamma: Dry density
        econ = 3.35 * 10**4 * ( gamma/24.0 )**(2) * (fc/60.0)**(1.0/3.0)
        return econ

    #
    # Ra: Rebar Area mm2
    def Ra(self,index):
        if index =='D10':
            return 71.
        elif index =='D13':
            return 127.0
        elif index =='D16':
            return 199.0
        elif index =='D19':
            return 287.0
        elif index =='D22':
            return 387.0
        elif index =='D25':
            return 507.0
        elif index =='D29':
            return 642.0
        elif index =='D32':
            return 794.0
        elif index =='D35':
            return 957.0
        elif index =='D38':
            return 1140.
        elif index =='D41':
            return 1340.0
        #
        elif index=='D10+D13':
            return 99.0
        elif index=='D13+D16':
            return 163.0
        elif index=='D16+D19':
            return 243.0
        else:
            print("error")
            return 'Err.'

    #
    # Dia: Diameter mm
    def Dia(self,index):
        if index =='D10':
            return 10.0
        elif index =='D13':
            return 13.0
        elif index =='D16':
            return 16.0
        elif index =='D19':
            return 19.0
        elif index =='D22':
            return 22.0
        elif index =='D25':
            return 25.0
        elif index =='D29':
            return 29.0
        elif index =='D32':
            return 32.0
        elif index =='D35':
            return 35.0
        elif index =='D38':
            return 38.0
        elif index =='D41':
            return 41.0
        #
        elif index=='D10+D13':
            return 99.0
        elif index=='D13+D16':
            return 163.0
        elif index=='D16+D19':
            return 243.0
        else:
            print("error")
            return 'Err.'

    # mm2/m
    def Ra_p(self,index,p):
        # p: Bar pitch
        return self.Ra(index)*1000.0/p

# test
"""
obj = Aij_rc_set()
print( obj.Ra('D41'))
print( obj.Ra_p('D13',200.0))
"""
