import pandas as pd
import numpy as np
from scipy import interpolate
import os
from perc import wp
import re

def get_ini_year(_country):
    _f = read_csv(os.getcwd()+'/GMD_skims/'+_country+'.csv')
    try: iy = int(_f['year'].mean())
    except: iy = _f['year'][0]
    return(iy)

def some(rawdataframe, n):
    return rawdataframe.loc[np.random.choice(rawdataframe.index, n)]
	
def split_big_dframe(finalhhframe,hhcat):
    a=finalhhframe[['weight','reg02']].groupby('reg02').apply(lambda x:x['weight'].count())
    b=pd.DataFrame(a,columns=['count'])
    c=b.sort(columns=['count'],ascending=False)
    bool1 = c.cumsum()<c.cumsum()['count'].iloc[-1]/2
    list_reg = list(c[bool1].dropna().index)
    finalhhframe1=finalhhframe.loc[finalhhframe['reg02'].isin(list_reg),:]
    finalhhframe2=finalhhframe.loc[~finalhhframe['reg02'].isin(list_reg),:]
    int_columns=['children','old','decile']+['cat{}workers'.format(thecat) for thecat in hhcat['hhcat'].unique()]
    finalhhframe1=merges_rows_bis(int_columns,finalhhframe1)
    finalhhframe2=merges_rows_bis(int_columns,finalhhframe2)
    return finalhhframe1,finalhhframe2
	
def convert_int_to_float(rawdataframe):
    if rawdataframe["idh"].dtype=='O':
        rawdataframe[['wgthh2007','Y']]=rawdataframe[['wgthh2007','Y']].astype(float)
    else:
        rawdataframe[['idh','wgthh2007','Y']]=rawdataframe[['idh','wgthh2007','Y']].astype(float)
    return rawdataframe
	
def del_missing_Y(rawdataframe):
    "delete rows with missing income and distribute weights"
    rawdataframe=rawdataframe.drop(rawdataframe.loc[pd.isnull(rawdataframe["wgthh2007"]),:].index)
    missing=pd.isnull(rawdataframe["Y"])
    add2wgt=rawdataframe.loc[missing,'wgthh2007'].sum()
    scal=rawdataframe['wgthh2007'].sum()/(rawdataframe['wgthh2007'].sum()-add2wgt)
    rawdataframe=rawdataframe.drop(rawdataframe.loc[missing,:].index)
    rawdataframe['wgthh2007']=rawdataframe['wgthh2007']*scal

    try: rawdataframe['skilled'].fillna(0, inplace=True)
    except: 
        rawdataframe['skilled'] = rawdataframe['skilled'].cat.add_categories(0)
        rawdataframe['skilled'].fillna(0, inplace=True)

    rawdataframe['skilled']=rawdataframe['skilled'].astype(bool)
    return rawdataframe
	
def convert_to_str_if_poss(x):
    try: return str(int(float(x)))
    except: return x

def get_industry_dict(industry_list,gmd_ind_dtype):

    industry_list = industry_list.loc[industry_list['dtype'] == gmd_ind_dtype].drop('dtype',axis=1)
    industry_list['indata'] = industry_list['indata'].astype(gmd_ind_dtype)
    industry_dict = industry_list.set_index('indata').to_dict()['industrycode']
    return industry_dict

def update_industry_dict(df,column='industry'):
    tmp = df.reset_index().set_index(column)
    tmp = tmp.loc[~tmp.index.duplicated(keep='first'),:]
    tmp = tmp.reset_index()
    tmp[column].to_csv('~/Desktop/tmp/industries.csv')

def get_SILC_employment_file(countrycode,mf,rawdataframe):

    for _ in os.listdir(mf.data_silc):

        if countrycode in _: 
            df = pd.read_stata(mf.data_silc+'/'+_)
            df['hhid'] = df['hhid'].astype(rawdataframe['hhid'].dtype)
            df['pid'] = df['pid'].astype(rawdataframe['pid'].dtype)
            df = df.reset_index(drop=True).set_index(['hhid','pid'])
            rawdataframe = pd.merge(rawdataframe,df,left_on=['hhid','pid'],right_index=True).rename(columns={'pl111':'industry'})
            return rawdataframe

def find_indus(countrycode,rawdataframe,mf):

    # initialize
    has_sectoral_employment_data = True

    # look for input data
    if 'industrycat4' in rawdataframe.columns: rawdataframe.rename(columns={'industrycat4':'industry'},inplace=True)
    elif 'industrycat10' in rawdataframe.columns: rawdataframe.rename(columns={'industrycat10':'industry'},inplace=True)
    else: 
        try: 
            rawdataframe = get_SILC_employment_file(countrycode,mf,rawdataframe)
        except: 
            print('does not have employment data...abort')
            return None, False

    # get dictionary for translation to major sectors
    industry_dict = get_industry_dict(mf.industry_list,rawdataframe['industry'].dtypes)
        
    # set lstatus flag
    has_lstatus = 'lstatus' in rawdataframe.columns

    # update dictionary
    # update_industry_dict(rawdataframe['industry'])    
    rawdataframe['industry'].replace(industry_dict,inplace=True)


    if rawdataframe.dropna(how='any',subset=['industry']).shape[0] == 0: 
        has_sectoral_employment_data = False
        print('this country survey does not have employment data!') 
    return rawdataframe, has_sectoral_employment_data
	
def create_age_col(rawdataframe):
    "sort children from adults and old people"
    try: rawdataframe['age'] = rawdataframe['age'].astype('int')
    except: rawdataframe['age'] = rawdataframe['age'].replace('80 or over',80).fillna(0).astype('int')
    rawdataframe['isold']=(rawdataframe['age']>64)
    rawdataframe['isanadult']=(rawdataframe['age']>14)&(rawdataframe['age']<65)
    rawdataframe['isachild']=(rawdataframe['age']<15)
    return rawdataframe
	
def create_gender_col(rawdataframe):
    "identifies gender for adults only"
    rawdataframe['isawoman']=((rawdataframe['isanadult'])&(rawdataframe['gender']=='Female'))
    rawdataframe['isaman']=((rawdataframe['isanadult'])&(rawdataframe['gender']=='Male'))
    return rawdataframe

def create_urban_col(rawdataframe):
    "identifies gender for adults only"
    rawdataframe['isurban']=rawdataframe['urban']
    return rawdataframe  

def associate_indus_to_head(rawdataframe,hhcat):
    #sort by head to be able to have the spouse first when the head does not have an industry
    rawdataframe=rawdataframe.sort(columns=['idh','head'],ascending=False)
    #solve pbs with duplicate head of household or missing head
    checkheads=rawdataframe.loc[:,['idh','ishead']].groupby('idh',sort=False).apply(lambda x:x['ishead'].sum())
    #if no head, replaced by spouse first and then adult
    subset=rawdataframe.loc[(rawdataframe['idh'].isin(checkheads.loc[checkheads==0].index))&(1-rawdataframe['isachild']),['head','idh','ishead']]
    hop=subset.groupby('idh')
    rawdataframe.loc[rawdataframe['idh'].isin(hop['idh'].head(1).values),'ishead']=1
    #if more than one head, drop duplicates in the hh dataframe later.
    #associate a category to each person based on hhcat
    for group in rawdataframe.groupby(list(categories.values)):
        therow=(hhcat[list(categories.values)].values==np.array(group[0])).all(1)
        cat=hhcat.loc[therow,'hhcat'].values
        rawdataframe.loc[group[1].index,'headcat']=cat
    return rawdataframe
	
def indus_to_bool(rawdataframe,industringlist,useskill=False):
    for industring in industringlist:
        newstring='is'+industring
        rawdataframe[newstring]=(rawdataframe['isanadult']&(rawdataframe['industry']==industring))+0
    rawdataframe['noindustry']=(rawdataframe['isanadult']&pd.isnull(rawdataframe['industry']))+0
    return rawdataframe
	
def associate_cat2people(rawdataframe,hhcat):
    categories=hhcat.columns[1::]
    for group in rawdataframe.groupby(list(categories.values)):
        therow=(hhcat[list(categories.values)].values==np.array(group[0])).all(1)
        cat=hhcat.loc[therow,'hhcat'].values
        rawdataframe.loc[group[1].index,'cat']=int(cat)

    rawdataframe['cat']=rawdataframe['cat'].astype('int64')
    for thecat in hhcat['hhcat'].unique():
        catstring='iscat{}'.format(thecat)
        rawdataframe[catstring]=(rawdataframe['cat']==thecat)&(rawdataframe['isanadult'])
    return rawdataframe
	
def deal_with_head_issues(hhdataframe):
	#look for households where someone else than the head has an industry (but not the head)
	otherworkers=(hhdataframe['noindustry']==1)&((hhdataframe['agworkers']>0)|(hhdataframe['manuworkers']>0)|(hhdataframe['servworkers']>0))
	sub=rawdataframe.loc[(rawdataframe['idh'].isin(hhdataframe.loc[otherworkers,:].index))&(rawdataframe['isanadult'])&(rawdataframe['headcat']<7),['head','idh','headcat']].copy()
	#takes the category of the spouse or if not, the first member of the hh with an industry
	hop=sub.groupby('idh')
	newcats=hop['headcat'].head(1)
	theindexes=hop['idh'].head(1).values
	hhdataframe.loc[theindexes,'headcat']=newcats.values
	return hhdataframe

def correct_zeroY(rawdataframe):
	min_Y=rawdataframe.loc[rawdataframe['Y']>0,'Y'].min()
	rawdataframe.loc[rawdataframe['Y']==0,'Y']=min_Y
	return rawdataframe
	
def sumoverhh(hhdataframe,rawdataframe,hhcolstring,rawstring):
	hhdataframe.loc[hhdataframe.index,hhcolstring]=rawdataframe.loc[:,['idh',rawstring]].groupby('idh',sort=False).apply(lambda x:x[rawstring].sum())
	return hhdataframe
	
def intensify_cat_columns(hhdataframe,rawdataframe,hhcat):
    for thecat in hhcat['hhcat'].unique():        
        catstring='iscat{}'.format(thecat)
        intstring='cat{}workers'.format(thecat)
        hhdataframe=sumoverhh(hhdataframe,rawdataframe,intstring,catstring)
    return hhdataframe
	
def match_deciles(hhdataframe,deciles):
    hhdataframe.loc[hhdataframe['Y']<=deciles[0],'decile']=1
    for j in np.arange(1,len(deciles)):
        hhdataframe.loc[(hhdataframe['Y']<=deciles[j])&(hhdataframe['Y']>deciles[j-1]),'decile']=j+1
    return hhdataframe
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data
	
def get_pop_description(countrycode,rawdataframe,mf,listofdeciles,issplit=False):

    # Extracts the three main components of our pb from the country dataframe: 
    # - 1) characteristics is a matrix that has all important household characteristics in columns and hh heads in lines. 
    # - 2) weights is the weight of each hh head 
    # - 3) pop_description is a summary of total population: number of children, skilled people etc"

    try: rawdataframe=rawdataframe.drop(rawdataframe.loc[pd.isnull(rawdataframe["wgthh2007"]),:].index).dropna(how='all')
    except: rawdataframe=rawdataframe.drop(rawdataframe.loc[pd.isnull(rawdataframe['wgt']),:].index).dropna(how='all')

    rawdataframe=convert_int_to_float(rawdataframe)
    rawdataframe=del_missing_Y(rawdataframe)
    rawdataframe=correct_zeroY(rawdataframe)

    # head of household info
    rawdataframe.to_csv('~/Desktop/tmp/{}.csv'.format(countrycode))
    try: rawdataframe['ishead']=rawdataframe['head']=="Head of household"
    except: 
        try: rawdataframe['ishead']=rawdataframe['relationharm']==1
        except: 
            rawdataframe = rawdataframe.reset_index(drop=True).set_index('hhid')
            rawdataframe['ishead'] = False
            rawdataframe.loc[~(rawdataframe.index.duplicated(keep='first')),'ishead'] = True
            rawdataframe = rawdataframe.reset_index()

    # get industry info
    rawdataframe,has_sectoral_employment_data=find_indus(countrycode,rawdataframe,mf)
    if not has_sectoral_employment_data: return None, has_sectoral_employment_data

    rawdataframe=create_age_col(rawdataframe)
    rawdataframe=create_urban_col(rawdataframe)

    rawdataframe=indus_to_bool(rawdataframe,['serv','manu','ag'])
    rawdataframe=associate_cat2people(rawdataframe,mf.hhcat)

    # rawdataframe['isskillworker']=rawdataframe['skill']&(rawdataframe['isanadult'])&(~rawdataframe['noindustry'])
    # rawdataframe=associate_indus_to_head(rawdataframe,mf.hhcat)
    #keep households instead of people.
    hhdataframe=rawdataframe.groupby('idh').apply(lambda x:x.head(1))
    hhdataframe.index=hhdataframe['idh'].values
    hhdataframe.loc[hhdataframe.index,'totY']=rawdataframe.loc[:,['idh','Y']].groupby('idh',sort=False).apply(lambda x:x['Y'].sum())
    # hhdataframe['meanY']=hhdataframe['Y']
    #deletes rows corresponding to the same household
    hhdataframe=hhdataframe.drop_duplicates(['idh'])
    #calculate the number of children, adults and old people in each hh

    hhdataframe=sumoverhh(hhdataframe,rawdataframe,'totwgt','wgthh2007')
    hhdataframe=sumoverhh(hhdataframe,rawdataframe,'children','isachild')
    hhdataframe=sumoverhh(hhdataframe,rawdataframe,'adults','isanadult')
    hhdataframe=sumoverhh(hhdataframe,rawdataframe,'old','isold')
    hhdataframe=sumoverhh(hhdataframe,rawdataframe,'urban','isurban')

    #calculate the number of workers in each category defined in mf.hhcat
    hhdataframe=intensify_cat_columns(hhdataframe,rawdataframe,mf.hhcat)	
    # hhdataframe=deal_with_head_issues(hhdataframe)
    #calculates the decile of each hh and adds a column
    deciles=wp(reshape_data(hhdataframe['Y']),reshape_data(hhdataframe['totwgt']),listofdeciles,cum=False)
    hhdataframe=match_deciles(hhdataframe,deciles)

    #columns to keep for description of population
    if not issplit:
        int_columns=['children','old','urban','decile']+['cat{}workers'.format(thecat) for thecat in mf.hhcat['hhcat'].unique()]
        finalhhframe=merges_rows(int_columns,hhdataframe)
    else:
        int_columns=['idh','Y','totY','reg02','children','old','decile','urban']+['cat{}workers'.format(thecat) for thecat in mf.hhcat['hhcat'].unique()]
        finalhhframe=hhdataframe[int_columns]
        finalhhframe['totweight']=hhdataframe['totwgt']
        finalhhframe['weight']=hhdataframe['wgthh2007']
        finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
        finalhhframe['nbpeople'].fillna(0, inplace=True)
    try:finalhhframe = finalhhframe.drop('totweight')
    except: pass

    return finalhhframe, has_sectoral_employment_data
	
def merges_rows(int_columns,hhdataframe):
    
    "merges the rows that have the same characteristics for the int_columns variables, to reduce the number of households. Weights are summed and I take the mean income between similar households"
    inter_wh=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totwgt'].sum())
    inter_w=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['wgthh2007'].sum())
    indexes=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['idh'].head(1))
    inter_c=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x[int_columns].head(1))
    inter_it=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totY'].mean())
    inter_i=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['Y'].mean())
    finalhhframe=pd.DataFrame(inter_c.values,columns=int_columns,index=indexes.values)
    # finalhhframe.drop('decile', axis=1, inplace=True)
    finalhhframe['totweight']=inter_wh.values
    finalhhframe['weight']=inter_w.values
    finalhhframe['totY']=inter_it.values
    finalhhframe['Y']=inter_i.values
    finalhhframe['idh']=indexes.values
    finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
    return finalhhframe
	
def merges_rows_bis(int_columns,hhdataframe):
    "merges the rows that have the same characteristics for the int_columns variables, to reduce the number of households. Weights are summed and I take the mean income between similar households"
    inter_wh=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totweight'].sum())
    inter_w=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['weight'].sum())
    indexes=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['idh'].head(1))
    inter_c=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x[int_columns].head(1))
    inter_it=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totY'].mean())
    inter_i=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['Y'].mean())
    finalhhframe=pd.DataFrame(inter_c.values,columns=int_columns,index=indexes.values)
    # finalhhframe.drop('decile', axis=1, inplace=True)
    finalhhframe['totweight']=inter_wh.values
    finalhhframe['weight']=inter_w.values
    finalhhframe['totY']=inter_it.values
    finalhhframe['Y']=inter_i.values
    finalhhframe['idh']=indexes.values
    finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
    return finalhhframe

def get_pop_data_from_UN(UNpop,countrycode,theyear):
    "get the description of the country's population in the projected year from UN/WB data"
    year='YR'+str(theyear)
    select=(UNpop['Country_Code']==countrycode)&((UNpop['Time_ValueCode']==year))
    country_pop=UNpop.loc[select,:].pivot(index='Time_ValueCode',columns='Indicator_Code',values='Value')
    pop_tot=country_pop.loc[year,'SP.POP.TOTL']
    pop_0014=country_pop.loc[year,'SP.POP.0014.TO']
    pop_1564=country_pop.loc[year,'SP.POP.1564.TO']
    pop_65up=country_pop.loc[year,'SP.POP.65UP.TO']
    return pop_tot,pop_0014,pop_1564,pop_65up

def get_pop_data_from_ssp_by_region(ssp,year):

    income_classification = read_csv('wbccodes2014.csv').set_index('country')[['wbregionname']]#.replace('High income: OECD','High income')

    ssp_data = read_csv('ssp_data/SspDb_country_data_2013-06-12.csv',low_memory=False)

    model='IIASA-WiC POP'
    if ssp==4: ssp='4d'
    scenario="SSP{}_v9_130115".format(ssp)
    #
    selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)
    pop_tot=ssp_data.loc[selection&(ssp_data['VARIABLE']=="Population"),['REGION',str(year)]].rename(columns={'REGION':'country'})
    pop_tot = merge(pop_tot.reset_index(),income_classification.reset_index(),on='country').set_index('wbregionname')[str(year)]
    #
    reg_tot = pop_tot.sum(level='wbregionname')*1E6
    return reg_tot


def get_pop_data_from_ssp(ssp_data,ssp,year,countrycode,_global=False):

    model='IIASA-WiC POP'
    if ssp==4: ssp='4d'
    scenario="SSP{}_v9_130115".format(ssp)
    if not _global: 
        selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)&(ssp_data['REGION']==countrycode)
        pop_tot=ssp_data.loc[selection&(ssp_data['VARIABLE']=="Population"),str(year)].squeeze()

    else:
        selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)
        pop_tot=ssp_data.loc[selection&(ssp_data['VARIABLE']=="Population"),str(year)].sum().squeeze()        

    pop_0014=0
    pop_1564=0
    pop_65up=0
    for gender in ['Male','Female']:
        for age in ['0-4','5-9','10-14']:
            var="Population|{}|Aged{}".format(gender,age)
            pop_0014+=ssp_data.loc[selection&(ssp_data['VARIABLE']==var),str(year)].values.sum()
        for age in ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64']:
            var="Population|{}|Aged{}".format(gender,age)
            pop_1564+=ssp_data.loc[selection&(ssp_data['VARIABLE']==var),str(year)].values.sum()
        for age in ['65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+']:
            var="Population|{}|Aged{}".format(gender,age)
            pop_65up+=ssp_data.loc[selection&(ssp_data['VARIABLE']==var),str(year)].values.sum()

    skilled_adults=0
    for gender in ['Male','Female']:
        for age in ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64']:
            for edu in ['Secondary Education','Tertiary Education']:
                var="Population|{}|Aged{}|{}".format(gender,age,edu)
                skilled_adults+=ssp_data.loc[selection&(ssp_data['VARIABLE']==var),str(year)].values.sum()
    pop_tot=pop_tot*10**6
    pop_0014=pop_0014*10**6
    pop_1564=pop_1564*10**6
    pop_65up=pop_65up*10**6
    skilled_adults=skilled_adults*10**6
    return pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults
	
def rescale_to_2007(UNpop,countrycode,year,characteristics,weight,income):
    "rescales the population to match 2007 description but keep aggregate income constant"
    pop_description=calc_pop_desc(characteristics,weight)
    pop_tot,pop_0014,pop_1564,pop_65up=get_pop_data_from_UN(UNpop,countrycode,year)
    ini_pop_desc=pop_description[['children','adults','old']]
    ini_pop_desc['income']=GDP(income,weight)
    new_pop_desc=ini_pop_desc.copy()
    new_pop_desc[['children','adults','old']]=[pop_0014,pop_1564,pop_65up]
    charac=characteristics[['children','adults','old']]
    charac['income']=income
    new_weights,result=build_new_weights(ini_pop_desc,new_pop_desc,charac,weight,ismosek=True)
    return new_weights

def country2r32(codes_tables,countrycode):
    r32='R32{}'.format(codes_tables.loc[codes_tables['ISO']==countrycode,'R32'].values[0])
    return r32

def r32_to_country(countrycode):
    codes_tables = read_csv('ssp_data/ISO3166_and_R32.csv')
    countrycode='{}'.format(codes_tables.loc[codes_tables['R32']==countrycode,'ISO'].values[0])
    return countrycode

def get_gdp_growth(ssp_data,year,ssp,r32,ini_year):

    #model='OECD Env-Growth'
    model='IIASA GDP'
    scenario="SSP{}_v9_130219".format(ssp)
    selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)&(ssp_data['REGION']==r32)&(ssp_data['VARIABLE']=='GDP|PPP')

    if ini_year<2010:
        y1=ssp_data.loc[selection,'2005'].values[0]
        y2=ssp_data.loc[selection,'2010'].values[0]
        f=interpolate.interp1d([2005,2010], [y1,y2],kind='slinear')
        gdp_ini=f(ini_year)
    else:
        y1=ssp_data.loc[selection,'2010'].values[0]
        try: 
            y2=ssp_data.loc[selection,'2015'].values[0]
            f=interpolate.interp1d([2010,2015], [y1,y2],kind='slinear')
            gdp_ini=f(ini_year)   
        except:
            y2=ssp_data.loc[selection,'2020'].values[0]
            f=interpolate.interp1d([2010,2020], [y1,y2],kind='slinear')
            gdp_ini=f(ini_year)   
         
    gdp_growth=ssp_data.loc[selection,str(year)].values[0]/gdp_ini
    return gdp_growth


def get_ppp_factors(countrycode,x,ppp_df):
    result = float(ppp_df.loc[ppp_df.datalevel==x.datalevel,['cpi2011','icp2011']].prod(axis=1))**(-1)
    return result

def create_correct_data(mf,countrycode,issplit=False):

    path = mf.data_gmd_raw+mf.country_file_dict[countrycode]
    rawdataframe=pd.read_stata(path,convert_categoricals=False)

    # rawdataframe.head(10).to_csv('~/Desktop/tmp/{}.csv'.format(countrycode))

    # datalevel is used for PPP weighting
    if (countrycode != 'IND' and countrycode != 'IDN' and countrycode != 'CHN'): rawdataframe['datalevel'] = 2

    # get weight
    if 'weight' not in rawdataframe.columns: rawdataframe = rawdataframe.rename(columns={'weight_h':'weight'})

    # get education level
    has_skill = True
    try: rawdataframe['skilled'] = rawdataframe['educy'].copy()
    except:
        try: rawdataframe['skilled'] = rawdataframe['educat4'].copy()
        except: has_skill = False
            
    # set welfare (annual per capita income/consumption in LCU) -> daily ppp
    # rawdataframe['Y'] = welfare/cpi2011/icp2011/365
    rawdataframe['Y'] = rawdataframe.eval('welfare/cpi2011/icp2011/365')

    # if use_minh_file:
        # conv_df = pd.read_stata('finalhhdataframes/_Final_CPI_PPP_to_be_used.dta')[['code','year','datalevel','cpi2011','icp2011']]
        # conv_df = conv_df.loc[(conv_df.code==countrycode)&(conv_df.year==rawdataframe.year.mean())]
        # ppp = rawdataframe.apply(lambda x:get_ppp_factors(countrycode,x,conv_df),axis=1)
        # rawdataframe['Y'] = ppp*rawdataframe['welfare']

        #ppp_df.to_csv('~/Desktop/tmp/ppp.csv')
        #ppp_df = ppp_df.loc[(ppp_df.code==countrycode)&(ppp_df.datalevel==df_datalevel)&(ppp_df.year==rawdataframe.year.mean())]
        #rawdataframe['Y'] = rawdataframe['welfare']/ppp_df[['cpi2011','icp2011']].prod(axis=1).squeeze()
        
    for essential_col in [['wgthh2007','weight',0],
                          ['idh','hhid',-1]]:

        if essential_col[0] not in rawdataframe.columns:
            rawdataframe[essential_col[0]] = rawdataframe[essential_col[1]].copy().fillna(essential_col[2])


    listofdeciles=np.sort(np.append(np.arange(0.1, 1.1, 0.1),[0.99]))
    if issplit:
        finalhhframe, has_sectoral_employment_data = get_pop_description(countrycode,rawdataframe,mf,listofdeciles,issplit=True)
    else:
        finalhhframe, has_sectoral_employment_data = get_pop_description(countrycode,rawdataframe,mf,listofdeciles)
    
    if finalhhframe is not None:
        if finalhhframe["idh"].dtype=='O':
            try: 
                finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
                finalhhframe["idh"]=finalhhframe["idh"].astype(float)
                finalhhframe.index=finalhhframe['idh']
            except: pass

    failure_types = (has_sectoral_employment_data,has_skill)
    return finalhhframe,failure_types
	
def filter_country(countrycode,all_surveys,codes):

    if countrycode in ['SSD','KIR','TUV','AFG']:
        return None

    if len(codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'])>0:
        wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
        if wbreg == 'YHI': wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'iiasaregion'].values[0]
    else:
        return None

    #if (countrycode=='COL')|(countrycode=='IND'):
    #    toobig=True
    #else:
    toobig=False
    finalhhframe=load_correct_data(all_surveys[countrycode])
    countrycode=correct_countrycode(countrycode)
    return finalhhframe,countrycode,toobig,wbreg
		
def load_correct_data(finalhhframe):
    if finalhhframe["idh"].dtype=='O':
        try: 
            finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
            finalhhframe["idh"]=finalhhframe["idh"].astype(float)
        except: pass
        finalhhframe.index=finalhhframe['idh']
        
    if 'hhweights' in finalhhframe.columns:
        finalhhframe.rename(columns={'hhweights':'weight'},inplace=True)
    return finalhhframe

		
def get_scenario_dataframe(outputs,countrycode,year,scenarname):
    finalhhframe = read_csv(outputs+"futurehhframe{}_{}_{}.csv".format(countrycode,year,scenarname))
    if (countrycode=='COL')|(countrycode=='IND'):
        toobig=True
    else:
        toobig=False
    return finalhhframe,countrycode,toobig
	
def correct_countrycode(countrycode):
    '''
    Corrects countrycodes in the database that don't correspond to official 3 letters codes.
    '''
    if countrycode=='TMP':
        countrycode='TLS'
    if countrycode=='ZAR':
        countrycode='COD'
    if countrycode=='ROM':
        countrycode='ROU'
    return countrycode

def reverse_correct_countrycode(countrycode):
    '''
    Corrects countrycodes in the database that don't correspond to official 3 letters codes.
    '''
    if countrycode=='TLS':
        countrycode='TMP'
    if countrycode=='COD':
        countrycode='ZAR'
    if countrycode=='ROU':
        countrycode='ROM'
    return countrycode
