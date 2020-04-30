import sys
from pandas import Series,DataFrame,read_csv,read_stata
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

data2day      = 30
year          = 2030
#ini_year      = 2007

nameofthisround = 'spring2020'

model             = os.getcwd()
data              = model+'/data/'
scenar_folder     = model+'/scenar_def/'
finalhhdataframes = model+'/finalhhdataframes_new/'
ssp_folder        = model+'/ssp_data/'
pik_data          = model+'/pik_data/'
iiasa_data        = model+'/iiasa_data/'

codes         = read_csv('wbccodes2014.csv')
codes_tables  = read_csv(ssp_folder+"ISO3166_and_R32.csv")
ssp_pop       = read_csv(ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
ssp_gdp       = read_csv(ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
hhcat         = read_csv(scenar_folder+"hhcat_v2.csv")
industry_list = read_csv(scenar_folder+"list_industries.csv")

dataset = 'GMD'#I2D2'#'GIDD'#'


#if dataset == 'GIDD':
	#data_gidd_csv     = model+'/data_gidd_csv_v4/'
	#list_csv=os.listdir(data_gidd_csv)
	#list_countries  = [re.search('(.*)_GIDD.csv', s).group(1) for s in list_csv]

	#for countrycode in list_countries:
		# if not os.path.isfile("finalhhdataframes/"+countrycode+"_finalhhframe.csv"):
		#wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
		#if wbreg == 'YHI':
			#continue
		#finalhhframe = create_correct_data(countrycode,data_gidd_csv,hhcat,industry_list)
		#if finalhhframe["idh"].dtype=='O':
			#finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
			#finalhhframe["idh"]=finalhhframe["idh"].astype(float)
		#finalhhframe.to_csv("finalhhdataframes/"+countrycode+"_finalhhframe.csv",encoding='utf-8',index=False)



#elif dataset == 'I2D2':
	
	# This bit creates the country skims in I2D2_skims/
	#if True:
		
		#data_i2d2_dta = model+'/thumbI2D2/'
		#list_csv=os.listdir(data_i2d2_dta)
		#print(list_csv)
		#list_files  = [re.search('(.*).dta', s).group(1) for s in list_csv if 'dta' in s]
		#list_files = ['ALL IN ONE REV 6_MENA']

		#for _f in list_files:
			#print(_f)
			# do this thing to make sure it's not a stata issue:
			#read_stata(data_i2d2_dta+_f+'.dta',convert_categoricals=False).head(1000000).to_csv('~/Desktop/tmp/check.csv')

			#reader=read_stata(data_i2d2_dta+_f+'.dta',chunksize=200000,convert_categoricals=False)
			#
			#df = DataFrame()
			#for nitm,itm in enumerate(reader):
				#df=df.append(itm.drop('rprevious',axis=1))

				# when I have 2 countries in the cache, write out the first one
				#countries_in_cache = df.ccode.unique()
				#print(nitm,countries_in_cache,df.year.max())
				#if len(countries_in_cache) > 1:
					#_df = df.loc[df.ccode==countries_in_cache[0]].copy()
					#_dfyrmax = _df.dropna(subset=['wage']).year.max()
					#_df.loc[_df.year==_dfyrmax].to_csv(model+'/I2D2_skims/'+countries_in_cache[0]+'_'+str(_dfyrmax)+'.csv')
					#df = df.loc[(df.ccode!=countries_in_cache[0])]
					#print(countries_in_cache[0])
				# cut on the year, but only if there's one country + more than one year in the cache
				#elif len(countries_in_cache) == 1 and len(df.year.unique())>1: df = df.loc[df.year==df.year.max()]
					  
			#df.to_csv(model+'/I2D2_skims/'+countries_in_cache[0]+'.csv')
			# gets the last country, which won't get written out by above code


	# This loads from I2D2_skims and tries to feed into Julie's function
	#data_i2d2_skims = model+'/I2D2_skims/'
	#list_csv=os.listdir(data_i2d2_skims)
	#list_countries  = [s.replace('.csv','') for s in list_csv]

	#for countrycode in list_countries:
		
		# if not os.path.isfile("finalhhdataframes/"+countrycode+"_finalhhframe.csv"):
		#try: wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
		#except: wbreg = None

		#if wbreg == 'YHI':
			#continue
		#finalhhframe = create_correct_data(countrycode,data_i2d2_skims,hhcat,industry_list,dataset='I2D2').dropna(how='all',axis=1)
		#if finalhhframe["idh"].dtype=='O':
			#finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
			#finalhhframe["idh"]=finalhhframe["idh"].astype(float)
		#finalhhframe.to_csv("finalhhdataframes_I2D2/"+countrycode+"_finalhhframe.csv",encoding='utf-8',index=False)

#elif dataset == 'GMD':
if dataset == 'GMD':
    # Look at the files
    if True:
        data_gmd_dta = model+'/GMD/GMD/'
        list_dta = []
        list_dta=os.listdir(data_gmd_dta)
        #list_countries = [_ for _ in os.listdir(data_gmd_dta) if 'DS_Store' not in _]
        for dta in range(len(list_dta)):
            dta_code = list_dta[dta][0]+list_dta[dta][1]+list_dta[dta][2]
            if not os.path.isfile("GMD_skims/"+dta_code+".csv"):
                read_stata(data_gmd_dta+list_dta[dta]).to_csv(model+'/GMD_skims/'+dta_code+'.csv')
                print('\n',dta_code)
            else:
                print(dta_code,'skim exists')
                
        #for _country in list_countries:
            #if not os.path.isfile("GMD_skims/"+_country+".csv"):
                #read_stata(data_gmd_dta+_country).to_csv(model+'/GMD_skims/'+_country+'.csv')
                #print('\n',_country)
            #else:
                #print(_country,'skim exists')
                
			#for _dir in [_ for _ in os.listdir(data_gmd_dta+'/'+_country) if 'DS_Store' not in _]:
				#for _subdir in  [_ for _ in os.listdir(data_gmd_dta+'/'+_country+'/'+_dir) if 'DS_Store' not in _]:
					#_path = data_gmd_dta+_country+'/'+_dir+'/'+_subdir+'/'
				
					#for __ in [_ for _ in os.listdir(_path) if 'DS_Store' not in _]:
						#if 'ALL' in __: read_stata(_path+'/'+__,convert_categoricals=False).to_csv(model+'/GMD_skims/'+_country+'.csv')
						#else: read_stata(_path+'/'+__).to_csv(model+'/GMD_skims/'+_country+__+'.csv')

    	# This loads from GMD_skims and tries to feed into Julie's function
        rich_countries = []#'AUT','BEL']
        countries_to_skip = []#['IND']#['ARG','BGD','BFA','BEN','AFG','BLR','BOL','BDI']#'BGR'
        
        data_gmd_skims = model+'/GMD_skims/'
        list_csv = []
        list_csv = os.listdir(data_gmd_skims)
        #list_countries  = [s.replace('.csv','') for s in list_csv if 'DS_Store' not in s]
        
        list_countries = []
        for skimname in range(len(list_csv)):
            skimcode = list_csv[skimname][0]+list_csv[skimname][1]+list_csv[skimname][2]
            list_countries.append(skimcode)
        
        #_record_bad = read_csv('results_summary/GMD_to_finalhhframe_status.csv').set_index('country')
        
        #for countrycode in ['MYS']:
        for countrycode in list_countries:
            wbreg = codes.loc[codes['country']==countrycode,'wbregion'].values
            try:
                wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values
            except:
                print('cant find wbreg for ',countrycode)
                
        for countrycode in list_countries:
            
            if countrycode in countries_to_skip: continue # hard-coded above
            #if countrycode in _record_bad.index and _record_bad.loc[countrycode,'skim_is_final']: continue
    		#if countrycode in _record_bad.index and _record_bad.loc[countrycode,'BAU_complete']: continue
    		#if countrycode in _record_bad.index and not _record_bad.loc[countrycode,'GMD_has_sectoral_employment_data']: continue
    		#if countrycode in _record_bad.index and not _record_bad.loc[countrycode,'GMD_has_skill_level']: continue
            else:
                print('\n--> running',countrycode)
                #_record_bad.loc[countrycode,['is_high_income',
                #'GMD_has_sectoral_employment_data',
                #'GMD_has_skill_level',
                #'skim_is_final']] = [False,False,None,None]
                
            if not os.path.isfile("finalhhdataframes/"+countrycode+"_finalhhframe.csv"):
                try: wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
                except:
                    print('cant find wbreg for ',countrycode)
                    #_record_bad.loc[countrycode,'skim_is_final'] = False
                    #_record_bad.sort_index().to_csv('results_summary/GMD_to_finalhhframe_status.csv')
                    continue

        	#if wbreg == 'YHI': _record_bad.loc[countrycode,'is_high_income'] = True
            #else: _record_bad.loc[countrycode,'is_high_income'] = False

            #try: 
            if True:
                finalhhframe, failure_types = create_correct_data(countrycode,data_gmd_skims,hhcat,industry_list,dataset='GMD')
                has_sectoral_employment_data,has_skill_level = failure_types
                #_record_bad.loc[countrycode,'GMD_has_sectoral_employment_data'] = has_sectoral_employment_data
                #_record_bad.loc[countrycode,'GMD_has_skill_level'] = has_skill_level
                
                if not has_sectoral_employment_data or not has_skill_level:
                    print('did not save out final df for '+countrycode)
                    #_record_bad.loc[countrycode,'skim_is_final'] = False
                    
                else:
                    if finalhhframe["idh"].dtype=='O':
                        try:
                            finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
                            finalhhframe["idh"]=finalhhframe["idh"].astype(float)
                        except: pass
                    
                    finalhhframe = finalhhframe.dropna(how='all',axis=1)
                    finalhhframe.to_csv("finalhhdataframes_GMD/"+countrycode+"_finalhhframe.csv",encoding='utf-8',index=False)
                    print('got final df for '+countrycode)
                    #_record_bad.loc[countrycode,['skim_is_final','BAU_complete','CC_complete']] = [True,False,False]
                    
			#except:
                #print('...multiple failures')
                #_record_bad.loc[countrycode,['skim_is_final','BAU_complete','CC_complete']] = [False,False,False]
                #_record_bad.sort_index().to_csv('results_summary/GMD_to_finalhhframe_status.csv')
