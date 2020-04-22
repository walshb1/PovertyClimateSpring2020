import pandas as pd
from pandas_helper import *
import numpy as np
import os

from lib_for_data_reading import get_pop_data_from_ssp_by_region, r32_to_country
#
is_folu = False
is_redx = True
#
nameofthisround = 'RedX' if is_redx else 'results' if is_folu else None
#

def build_results_df(folder,list,codes,set=pd.DataFrame()):
    income_classification = (pd.read_csv('wbccodes2014.csv').set_index('country')[['wbincomename']]
                             .replace('High income: OECD','High income')
                             .replace('High income: nonOECD','High income'))

    income_classification.loc['TLS'] = 'Low income'
    income_classification.loc['COD'] = 'Low income'
    income_classification = income_classification.to_dict()['wbincomename']


    #set['income_class'] = None
    for file in list:
        try: 
            _c = pd.read_csv(folder+file,skiprows=1).set_index('country').T
            _c['income_class'] = income_classification[_c.index[0]]
            
            set = set.append(_c)
        except: pass
       
    set.index.name = 'country'
    set = set.reset_index().dropna(axis=0,how='all')

    set['countryname'] = (set.country.replace(codes.set_index('country.1').country_name)
                          .replace(codes.set_index('country.2').country_name)
                          .replace(codes.set_index('country.3').country_name)
                          .replace(codes.set_index('country.4').country_name)
                          .replace(codes.set_index('country.5').country_name).replace(codes.set_index('country').country_name))
    
    set = set.dropna(how='all').set_index(['countryname', 'scenar', 'ssp','income_class'])
    return set

def correct_countrycode(countrycode):
    # Corrects countrycodes in the database that don't correspond to official 3 letters codes.
    if countrycode=='TMP': countrycode='TLS'
    if countrycode=='ZAR': countrycode='COD'
    if countrycode=='ROM': countrycode='ROU'
    return countrycode

def get_result(df,out_row,in_cols,label,_n,_b,_t,
               to_mil=True,is_avg=False,in_col_now=None):
    sf = (1E-6 if to_mil else (1E2 if 'gap' in out_row else 1))
    print(out_row,sf)
    _level = ['ssp','scenar']

    in_hack = in_col_now if in_col_now is not None else in_cols

    if is_redx:
        _scen = ((_t['switch_temp']=='True')&(_t['switch_disasters']=='True'))

        if not is_avg:
            df.loc[out_row] = [sf*float(_n[in_hack].prod(axis=1).sum(level=_level)[0]),                 # now
                               sf*float(_b[in_cols].prod(axis=1).sum(level=_level)[0]),                # base (2030)
                               sf*float(_t.loc[_scen,in_cols].prod(axis=1).sum(level=_level)[0]),                # transformation (2030)
                               sf*float((_t.loc[_scen,in_cols].prod(axis=1)-_b[in_cols].prod(axis=1)).sum(level=_level)[0]), # delta(tran-base)
                               label]
        else: 
            #print(sf*float(_b[in_cols].prod(axis=1).sum(level=_level)[0]))
            #print(sf*float(_t[in_cols].prod(axis=1).sum(level=_level)[0]))
            df.loc[out_row] = [sf*(float(_n[in_hack].prod(axis=1).sum(level=_level)[0])/_n[in_cols[0]].sum(level=_level)[0]),             # now
                               sf*(float(_b[in_cols].prod(axis=1).sum(level=_level)[0])/float(_b[in_cols[0]].sum(level=_level)[0])),   # base (2030)
                               sf*(float(_t.loc[_scen,in_cols].prod(axis=1).sum(level=_level)[0])/float(_t.loc[_scen,in_cols[0]].sum(level=_level)[0])), # transformation (2030)
                               sf*(float((_t.loc[_scen,in_cols].prod(axis=1)-_b[in_cols].prod(axis=1)).sum(level=_level)[0])/float(_b[in_cols[0]].sum(level=_level)[0])), # delta(tran-base)
                               label]
    
    if is_folu:
        if not is_avg:
            df.loc[out_row] = [sf*float(_n[in_hack].prod(axis=1).sum(level=_level).mean()),                 # now
                               sf*float(_b[in_cols].prod(axis=1).sum(level=_level)[0]),                # base (2030)
                               sf*float(_t[in_cols].prod(axis=1).sum(level=_level)[0]),                # transformation (2030)
                               sf*float((_t[in_cols].prod(axis=1)-_b[in_cols].prod(axis=1)).sum(level=_level)[0]), # delta(tran-base)
                               label]
        else: 
            df.loc[out_row] = [sf*float(_n[in_hack].prod(axis=1).sum(level=_level)/_n[in_cols[0]].sum(level=_level).mean()),             # now
                               sf*(float(_b[in_cols].prod(axis=1).sum(level=_level)[0])/float(_b[in_cols[0]].sum(level=_level)[0])),   # base (2030)
                               sf*(float(_t[in_cols].prod(axis=1).sum(level=_level)[0])/float(_t[in_cols[0]].sum(level=_level)[0])),   # transformation (2030)
                               sf*(float((_t[in_cols].prod(axis=1)-_b[in_cols].prod(axis=1)).sum(level=_level)[0])/float(_b[in_cols[0]].sum(level=_level)[0])), # delta(tran-base)
                               label]
        
    
    return df
    


# POVERTY HEADCOUNT @ $X.XX/day
def poverty_incidence(diff_poverty,now,base,tran,povline,do_by_sec=True):

    diff_poverty = get_result(diff_poverty,'poverty ('+povline+')',['pop_'+povline],'mil. persons',_n=now,_b=base,_t=tran)
    # sectors
    for _sec in ['_service','_agricul','_manufac']:
        diff_poverty = get_result(diff_poverty,'poverty ('+povline+') - '+_sec[1:],['pop_'+povline+_sec],'mil. persons',_n=now,_b=base,_t=tran)
    # locations
    for _loc in ['_urban','_rural']:
        diff_poverty = get_result(diff_poverty,'poverty ('+povline+') - '+_loc[1:],['pop_'+povline+_loc],'mil. persons',_n=now,_b=base,_t=tran)

    return diff_poverty



# POVERTY GAP @ Income of people in poverty, as % of poverty line
def poverty_gap(diff_poverty,now,base,tran,pline,do_by_sec=True):
    _str = 'poverty gap ('+pline+')'

    diff_poverty = get_result(diff_poverty,'poverty gap ('+pline+')',['pop_'+pline,'gap_'+pline],'hh in poverty: avg. income shortfall of poverty line (%)',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    if not do_by_sec: return diff_poverty

    for _sec in ['_service','_agricul','_manufac']:
        _str = 'poverty gap ('+pline+') - '+_sec[1:]
        _in_cols = ['pop_'+pline+_sec,'gap_'+pline+_sec]
        diff_poverty = get_result(diff_poverty,_str,_in_cols,'hh w/ '+_sec[1:]+' workers in poverty: avg. income shortfall of poverty line (%)',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)

    for _loc in ['_urban','_rural']:
        _str = 'poverty gap ('+pline+') - '+_loc[1:]
        _in_cols = ['pop_'+pline+_loc,'gap_'+pline+_loc]
        diff_poverty = get_result(diff_poverty,_str,_in_cols,'hh w/ '+_loc[1:]+' hh in poverty: avg. income shortfall of poverty line (%)',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
        
    return diff_poverty


def create_regional_output(year,now,base,tran,by_income=False):
    if not by_income: 
        
        _handle = 'wbregionname'

        regions = pd.read_csv('wbccodes2014.csv').set_index('country')[[_handle]]
        now = pd.merge(now.reset_index(),regions.reset_index(),on='country').set_index(_handle)
        base = pd.merge(base.reset_index(),regions.reset_index(),on='country').set_index(_handle)
        tran = pd.merge(tran.reset_index(),regions.reset_index(),on='country').set_index(_handle)

    else: 
        _handle = 'income_class'
        #
        income_classification = pd.read_csv('wbccodes2014.csv').set_index('country')[['wbincomename']].replace('High income: OECD','High income').replace('High income: nonOECD','High income')
        income_classification.loc['TLS'] = 'Low income'
        income_classification.loc['COD'] = 'Low income'
        
        now = pd.merge(now.reset_index(),income_classification.reset_index(),on='country').set_index(_handle)
        base = pd.merge(base.reset_index(),income_classification.reset_index(),on='country').set_index(_handle)
        tran = pd.merge(tran.reset_index(),income_classification.reset_index(),on='country').set_index(_handle)

    df = pd.DataFrame(columns=now.sum(level=_handle).index,index={})

    for _scen in['now','bau','tran']:
        #
        if _scen == 'now': _tmp = now.copy()
        elif _scen == 'bau': _tmp = base.copy()
        elif _scen == 'tran': _tmp = tran.copy()
        #
        _ix = np.array(_tmp.sum(level=_handle).index)
        #
        df.loc['total population {}'.format(_scen)] = None      
        df.loc['total population {}'.format(_scen)][_ix] = np.array(_tmp['tot_pop'].sum(level=_handle).squeeze().T)        
        #
        df.loc['unemployed {}'.format(_scen)] = None
        try: 
            try: df.loc['unemployed {}'.format(_scen)][_ix] = np.array(_tmp['unemployed'].sum(level=_handle).squeeze().T)    
            except: df.loc['unemployed {}'.format(_scen)][_ix] = np.array(_tmp['unemployed'].sum(level=_handle).iloc[0].squeeze().T) 
        except: pass
        #
        df.loc['urban decent jobs {}'.format(_scen)] = None
        df.loc['urban decent jobs {}'.format(_scen)][_ix] = np.array(_tmp['decent_jobs_urban'].sum(level=_handle).squeeze().T)
        #
        df.loc['rural decent jobs {}'.format(_scen)] = None
        df.loc['rural decent jobs {}'.format(_scen)][_ix] = np.array(_tmp['decent_jobs_rural'].sum(level=_handle).squeeze().T)
        #
        try: 
            df.loc['ppl_in_hh_working_poor_urban {}'.format(_scen)] = None
            df.loc['ppl_in_hh_working_poor_urban {}'.format(_scen)][_ix] = np.array(_tmp['decent_jobs_urban'].sum(level=_handle).squeeze().T)
            #
            df.loc['ppl_in_hh_working_poor_rural {}'.format(_scen)] = None
            df.loc['ppl_in_hh_working_poor_rural {}'.format(_scen)][_ix] = np.array(_tmp['decent_jobs_rural'].sum(level=_handle).squeeze().T)
        except: pass
        #
        df.loc['avg income Q1 {}'.format(_scen)] = None
        df.loc['avg income Q1 {}'.format(_scen)][_ix] = np.array((_tmp[['avg_income_bott20','tot_pop']].prod(axis=1).sum(level=_handle)
                                                                  /_tmp['tot_pop'].sum(level=_handle)).squeeze().T)
        df.loc['avg income all quintiles {}'.format(_scen)] = None
        df.loc['avg income all quintiles {}'.format(_scen)][_ix] = np.array((_tmp[['avg_income','tot_pop']].prod(axis=1).sum(level=_handle)
                                                                  /_tmp['tot_pop'].sum(level=_handle)).squeeze().T)

        df.loc['rural avg income Q1 {}'.format(_scen)] = None
        df.loc['rural avg income Q1 {}'.format(_scen)][_ix] = np.array((_tmp[['avg_income_bott20_rural','tot_pop']].prod(axis=1).sum(level=_handle)
                                                                        /_tmp['tot_pop'].sum(level=_handle)).squeeze().T)

        df.loc['rural avg income all quintiles {}'.format(_scen)] = None
        df.loc['rural avg income all quintiles {}'.format(_scen)][_ix] = np.array((_tmp[['avg_income_rural','tot_pop']].prod(axis=1).sum(level=_handle)
                                                                                   /_tmp['tot_pop'].sum(level=_handle)).squeeze().T)

        try: 
            df.loc['Q1 population exposed {}'.format(_scen)] = None
            df.loc['Q1 population exposed {}'.format(_scen)][_ix] = np.array(_tmp['exposed_pop_Q1'].sum(level=_handle).squeeze().T)
        except: pass

        for _pl in [190,320,550,1000]:
            #
            try:
                df.loc['population exposed {} ({})'.format(_scen,_pl)] = None
                df.loc['population exposed {} ({})'.format(_scen,_pl)][_ix] = np.array(_tmp['pop_{}'.format(_pl)].sum(level=_handle).squeeze().T)    
                #
                df.loc['poverty count {} ({})'.format(_scen,_pl)] = None
                df.loc['poverty count {} ({})'.format(_scen,_pl)][_ix] = np.array(_tmp['pop_{}'.format(_pl)].sum(level=_handle).squeeze().T)
                #
                df.loc['rural poverty count {} ({})'.format(_scen,_pl)] = None
                df.loc['rural poverty count {} ({})'.format(_scen,_pl)][_ix] = np.array(_tmp['pop_{}_rural'.format(_pl)].sum(level=_handle).squeeze().T)
                #
                df.loc['poverty gap {} ({})'.format(_scen,_pl)] = None
                df.loc['poverty gap {} ({})'.format(_scen,_pl)][_ix] = np.array((_tmp[['gap_{}'.format(_pl),'pop_{}'.format(_pl)]].prod(axis=1).sum(level=_handle)
                                                                                 /_tmp['pop_{}'.format(_pl)].sum(level=_handle)).squeeze().T)
                df.loc['rural poverty gap {} ({})'.format(_scen,_pl)] = None
                df.loc['rural poverty gap {} ({})'.format(_scen,_pl)][_ix] = np.array((_tmp[['gap_{}_rural'.format(_pl),'pop_{}_rural'.format(_pl)]].prod(axis=1).sum(level=_handle)
                                                                                       /_tmp['pop_{}_rural'.format(_pl)].sum(level=_handle)).squeeze().T)
            except: pass
            
    df['global'] = df.sum(axis=1)
    df.index.name = 'indicator'
    df.to_csv('{}_summary/by_region_{}_{}.csv'.format(nameofthisround,_handle,year))
    
    df = df.drop('global',axis=1)
    #

    # Globalize
    if not by_income:

        reg_pop_now = get_pop_data_from_ssp_by_region('2',2020).squeeze().to_dict()
        now_sf = {}
        for _c in df.columns:
            now_sf[_c] = reg_pop_now[_c]/df.loc['total population now',_c]

        reg_pop = get_pop_data_from_ssp_by_region('2',2030).squeeze().to_dict()
        model_sf = {}
        for _c in df.columns:
            model_sf[_c] = reg_pop[_c]/df.loc['total population bau',_c]

        for _sf in model_sf:
            for _scen in ['now','bau','tran']:
                for _c in ['total population {}'.format(_scen),
                           'unemployed {}'.format(_scen),
                           'urban decent jobs {}'.format(_scen),
                           'rural decent jobs {}'.format(_scen),
                           'ppl_in_hh_working_poor_urban {}'.format(_scen)]:
                    
                    try: df.loc[_c,_sf] *= (model_sf[_sf] if _scen!='now' else now_sf[_sf])
                    except: pass

                for _pl in [190,320,550]:
                    for _c in ['poverty count {} ({})'.format(_scen,_pl),
                               'rural poverty count {} ({})'.format(_scen,_pl)]:
                        df.loc[_c,_sf] *= (model_sf[_sf] if _scen!='now' else now_sf[_sf])
                    for _c in ['poverty gap {} ({})'.format(_scen,_pl),
                               'rural poverty gap {} ({})'.format(_scen,_pl)]:
                        df.loc[_c,_sf] *= 1E2
        # 
        df['global'] = df.sum(axis=1)

        for _scen in ['now','bau','tran']:      
            for _pl in [190,320,550]:
                
                for _col in ['poverty gap {} ({})'.format(_scen,_pl),
                             'rural poverty gap {} ({})'.format(_scen,_pl),
                             'avg income Q1 {}'.format(_scen,_pl),
                             'avg income all quintiles {}'.format(_scen,_pl),
                             'rural avg income Q1 {}'.format(_scen,_pl),
                             'rural avg income all quintiles {}'.format(_scen,_pl)]:
                    #
                    _num = 0; _dnm = 0
                    for _sf in model_sf:
                        _num += df.loc[_col,_sf]*(model_sf[_sf] if _scen!='now' else now_sf[_sf])  
                        _dnm += (model_sf[_sf] if _scen!='now' else now_sf[_sf])  
                    #
                    df.loc[_col,'global'] = _num/_dnm

        #df = df.drop([_i for _i in df.index if 'now' in _i],axis=0)
        df.sort_index().to_csv('{}_summary/by_region_globalized.csv'.format(nameofthisround))
    return True

    

def summarize_results_for_all_countries(year=2030):
    
    codes = pd.read_csv('wbccodes2014.csv')
    codes['country'] = codes.country.apply(correct_countrycode)
    codes['country.1'] = codes['country']+'.1'
    codes['country.2'] = codes['country']+'.2'
    codes['country.3'] = codes['country']+'.3'
    codes['country.4'] = codes['country']+'.4'
    codes['country.5'] = codes['country']+'.5'


    model = os.getcwd()
    now_folder   = "{}/{}_present/".format(model,nameofthisround)
    bau_folder   = "{}/{}_baselines{}/".format(model,nameofthisround,('_'+str(year) if year!=2030 else '')) 
    tr_folder    = "{}/{}_with_cc{}/".format(model,nameofthisround,('_'+str(year) if year!=2030 else ''))
    print('\n\n',bau_folder,'\n\n')

    #
    list_now = os.listdir(now_folder)
    list_bau = os.listdir(bau_folder)
    list_tr  = os.listdir(tr_folder)
    
    list_bau_noredist = (i for i in list_bau if 'redist' not in i)
    list_bau_redist = (i for i in list_bau if 'redist' in i)
    
    list_tr_noredist = (i for i in list_tr if 'redist' not in i)
    list_tr_redist  = (i for i in list_tr if 'redist' in i)
    #
    income_classification = pd.read_csv('wbccodes2014.csv').set_index('country')[['wbincomename']].replace('High income: OECD','High income').replace('High income: nonOECD','High income')
    income_classification = income_classification.to_dict()['wbincomename']
    income_classification['TLS'] = 'Low income'
    income_classification['COD'] = 'Low income'
    #

    now = build_results_df(now_folder,list_now,codes)
    base = build_results_df(bau_folder,list_bau_noredist,codes)
    tran = build_results_df(tr_folder,list_tr_noredist,codes)
    #
    dfGDP = now['GDP'].to_frame(name='now_GDP').astype('float')
    dfGDP['base_GDP'] = base['GDP'].astype('float')
    #
    dfGDP['now_pop'] = now['tot_pop'].astype('float')
    dfGDP['base_pop'] = base['tot_pop'].astype('float')    
    #
    dfGDP['now_GDPpc'] = dfGDP.eval('now_GDP/now_pop')
    dfGDP['base_GDPpc'] = dfGDP.eval('base_GDP/base_pop')
    #dfGDP['cc'] = tran['GDP']

    dfGDP.to_csv('~/Desktop/tmp/out_{}.csv'.format(year))

    #tran = tr.dropna().set_index(['countryname', 'scenar', 'ssp'])
    #base = broadcast_simple(bau.dropna().set_index(['countryname', 'scenar', 'ssp']),tran.index)
    #now  = broadcast_simple(now.dropna().set_index(['countryname', 'scenar', 'ssp']),tran.index)
    for df in [tran,base,now]: 
        for _c in df.columns:
            try: df[_c] = df[_c].astype('float')
            except: pass

    diff_poverty = 1E-6*base['tot_pop'].sum(level=['ssp','scenar']).to_frame(name='total pop')
    diff_poverty = diff_poverty.reset_index().T
    
    diff_poverty.columns = ['now']
    #diff_poverty.insert(0,'now')
    #assert(False)

    if is_folu:
        for _c in ['bau','transformation']:
            diff_poverty[_c] = None
        diff_poverty.loc['ssp',['now','bau','transformation']] = 'ssp2'

        diff_poverty.loc['year'] = ['see GMD survey',base.year.mean(),tran.year.mean()]
        
    elif is_redx:
        diff_poverty['base'] = None
        diff_poverty['base_with_cc'] = None
        diff_poverty.loc['ssp',['base','base_with_cc']] = ('ssp2','ssp2')
        
        diff_poverty.loc['year'] = ['see GMD survey',year,year]

    diff_poverty['delta (T-B)'] = None
    diff_poverty['units'] = None



    ###############################################
    diff_poverty = get_result(diff_poverty,'Total population',['tot_pop'],'mil.',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)  
    #
    diff_poverty = poverty_incidence(diff_poverty,now,base,tran,'190')
    diff_poverty = poverty_incidence(diff_poverty,now,base,tran,'320')
    diff_poverty = poverty_incidence(diff_poverty,now,base,tran,'550')

    #
    diff_poverty = poverty_gap(diff_poverty,now,base,tran,'190')
    diff_poverty = poverty_gap(diff_poverty,now,base,tran,'320')
    diff_poverty = poverty_gap(diff_poverty,now,base,tran,'550')

    ## By income grouping
    #create_regional_output(year,now,base,tran)
    create_regional_output(year,now,base,tran,by_income=True)
    #


    # INCOMES
    diff_poverty = get_result(diff_poverty,'Average income (pc)',['tot_pop','avg_income'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average urban income (pc)',['tot_pop','avg_income_urban'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average rural income (pc)',['tot_pop','avg_income_rural'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    #
    diff_poverty = get_result(diff_poverty,'Average agriculture income (pc)',['tot_pop','avg_income_ag'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average non-ag income (pc)',['tot_pop','avg_income_nonag'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    #
    diff_poverty = get_result(diff_poverty,'Poorest quintile income (pc)',['tot_pop','avg_income_bott20'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Poorest quintile & urban income (pc)',['tot_pop','avg_income_bott20_urban'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Poorest quintile & rural income (pc)',['tot_pop','avg_income_bott20_rural'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    #
    diff_poverty = get_result(diff_poverty,'Average income Q2 (pc)',['tot_pop','incQ2'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average income Q3 (pc)',['tot_pop','incQ3'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average income Q4 (pc)',['tot_pop','incQ4'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    diff_poverty = get_result(diff_poverty,'Average income Q5 (pc)',['tot_pop','incQ5'],'$ (ppp) per year',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)
    #
    #
    # HEADCOUNT
    diff_poverty = get_result(diff_poverty,'Children in agricultural hh',['childrenag'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)
    diff_poverty = get_result(diff_poverty,'Children in non-ag hh',['childrenonag'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)
    diff_poverty = get_result(diff_poverty,'People in agricultural hh',['peopleag'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)
    diff_poverty = get_result(diff_poverty,'People in non-ag hh',['peoplenonag'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)

    # HEADCOUNT
    try: 
        diff_poverty = get_result(diff_poverty,'Total in working poor hh urban',['ppl_in_hh_working_poor_urban'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)
        diff_poverty = get_result(diff_poverty,'Total in working poor hh rural',['ppl_in_hh_working_poor_rural'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False)
    except: pass

    # GINI
    diff_poverty = get_result(diff_poverty,'GINI',['tot_pop','gini'],'',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True)

    # URBAN
    diff_poverty = get_result(diff_poverty,'Urban pop.',['tot_pop','shareurban'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False,in_col_now=['urban'])
    diff_poverty = get_result(diff_poverty,'Urban fraction',['tot_pop','shareurban'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True,in_col_now=['urban'])  

    # EMPLOYMENT
    diff_poverty = get_result(diff_poverty,'Employed pop',['tot_pop','shareemp'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False,in_col_now=['employed']) 
    diff_poverty = get_result(diff_poverty,'Employed fraction',['tot_pop','shareemp'],'mil. persons',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True,in_col_now=['employed']) 

    # Disaster exposure
    if is_redx:
        for _ in ['fa20_earthquake','fa20_wind','fa20_surge','fa20_tsunami','fa20_riverflood']:
            diff_poverty = get_result(diff_poverty,'Avg. fraction affected by 20+ year '+_.replace('fa20_',''),['tot_pop',_],'mil. persons',_n=now,_b=base,_t=tran,to_mil=False,is_avg=True) 


    _dict = {'service_unskilled':'cat1workers', 'service_skilled':'cat2workers',
             'agricul_unskilled':'cat3workers', 'agricul_skilled':'cat4workers',
             'manufac_unskilled':'cat5workers', 'manufac_skilled':'cat6workers'}

    #############################################
    # NUMBER OF JOBS: for each (sector X skill)
    for _sec, _seccode in [('service','sv'),
                              ('agricul','ag'),
                              ('manufac','mn')]:
        for _ed in ['_unskilled','_skilled']:
            diff_poverty = get_result(diff_poverty,'jobs_'+_sec+_ed,['pop_'+_seccode+_ed],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False,in_col_now=[_dict[_sec+_ed]]) 

            for _loc in ['_urban','_rural']:
                diff_poverty = get_result(diff_poverty,'jobs_'+_sec+_ed+_loc,['pop_'+_seccode+_ed+_loc],'mil. persons',_n=now,_b=base,_t=tran,
                                          to_mil=True,is_avg=False,in_col_now=[_dict[_sec+_ed]+_loc]) 

    #############################################
    # DECENT JOBS (hh pc income > 1.90/day) for each sector
    for _sec in ['service','agricul','manufac']:
        diff_poverty = get_result(diff_poverty,'decent_jobs_'+_sec,['decent_jobs_'+_sec],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False) 

    for _loc in ['urban','rural']:
        diff_poverty = get_result(diff_poverty,'decent_jobs_'+_loc,['decent_jobs_'+_loc],'mil. persons',_n=now,_b=base,_t=tran,to_mil=True,is_avg=False) 



    #
    #if is_redx:
    #    diff_poverty = diff_poverty.drop(['delta (T-B)'],axis=1)

    diff_poverty.to_csv('{}_summary/diff_poverty'.format(nameofthisround)+('_redx' if is_redx else ('_FOLU' if is_folu else ''))+('_'+str(year) if year != 2030 else '')+'.csv')

summarize_results_for_all_countries()
try: summarize_results_for_all_countries(2050)
except: pass
