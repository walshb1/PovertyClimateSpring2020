import sys
from pandas import Series,DataFrame,read_csv
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re
import datetime

def run_multiprocessing(scenar,datalist,paramvar,all_surveys,switches,baselines,with_cc):
    for countrycode in list(all_surveys.keys()):
        if countrycode in ['IND','CHN','TUN']:
            continue
        if not os.path.isfile("{}forprim_cc_{}_b{}.csv".format(with_cc,countrycode,scenar)):
            print(countrycode)
            try:
                forprim_now,forprim_bau,forprim_cc, inc = country_run(countrycode,scenar,datalist,paramvar,all_surveys,switches,
                                                     b_value=0,future_with_cc=True, with_person_data=False)
                print(countrycode + ' model finished running')
                forprim_bau.to_csv("{}forprim_bau_{}_b{}.csv".format(baselines,countrycode,scenar))
                forprim_cc.to_csv("{}forprim_cc_{}_b{}.csv".format(with_cc,countrycode,scenar))
                print(countrycode + ' results saved')
            except:
                print(countrycode + ' not working')
                pass
            
    return 1