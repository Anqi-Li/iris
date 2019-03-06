#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:27:42 2019

@author: anqil
"""

import numpy as np
from db_functions import insert_osiris_db

orbit = 27031 
db_filename = '/home/anqil/Documents/osiris_database/OSIRIS_2006.db'
while orbit<=32400:
    insert_osiris_db(db_filename, orbit)
    #next orbit
    orbit += 1
