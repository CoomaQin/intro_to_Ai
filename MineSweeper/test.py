#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:26:44 2019

@author: ruixigu
"""

import random
import numpy as np
import game_board as gb
from alg import *
from draw import draw_board
from alg import TSMP, constraint_satisfaction_gauss, constraint_satisfaction


def test(dim,d,num):
    lose_count = 0
    total_density = 0
    for i in range(num):
        b = gb.Board(dim, d)
        #print(b.mine_list)
        m = DSSP(b)
        #m = TSMP(b,True,True)
        print(m)
        density = 1 - (int(str(b.boom)) / b.mine_num)
        print('density = ' + str(density))
        if int(str(b.boom)) >=1:
            print('lose')
            lose_count += 1
        print('the number of mines are queried: ' + str(b.boom))
        total_density += density
        score = total_density / num
    return (1- lose_count/num), score


print(test(10,10,20))

'''
b = gb.Board(10, 40)
m = TSMP(b, False, False)
print()
'''
