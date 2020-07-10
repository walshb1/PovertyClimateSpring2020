import pandas as pd
import sys, os, re

from lib_for_data_reading import *
from country_run import *
from lib_for_cc import *

data2day      = 30
year          = 2030
#ini_year      = 2007


class mainframe():

    def __init__(self,nameofthisround='spring2020'):

        # environment
        self.nameofthisround = nameofthisround

        # directories
        self.model             = os.getcwd()
        self.data              = self.model+'/data/'
        self.scenar_folder     = self.model+'/scenar_def/'
        self.finalhhdataframes = self.model+'/finalhhdataframes/'
        self.ssp_folder        = self.model+'/ssp_data/'
        self.pik_data          = self.model+'/pik_data/'
        self.iiasa_data        = self.model+'/iiasa_data/'

        # simulation inputs
        self.codes         = pd.read_csv('wbccodes2014.csv')
        self.codes_tables  = pd.read_csv(self.ssp_folder+"ISO3166_and_R32.csv")
        self.ssp_pop       = pd.read_csv(self.ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
        self.ssp_gdp       = pd.read_csv(self.ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
        self.hhcat         = pd.read_csv(self.scenar_folder+"hhcat_v2.csv")
        self.industry_list = pd.read_csv(self.scenar_folder+"list_industries.csv")

        # microdata from GMD
        if self.nameofthisround == 'spring2020_BW': 
            self.data_gmd = self.model+'/../GMD/'
            self.data_gmd_raw = self.model+'/../GMD/GMD2020/'
            self.data_silc = self.model+'/../GMD/SILC_employment/'
            self.data_gmd_skims = self.model+'/../GMD/skims/'
        else: 
            self.data_gmd_raw = self.model+'/../' # your directory structure, outside of GIT
            self.data_gmd_skims = self.model+'/../' # your directory structure, outside of GIT
        self.list_raw=os.listdir(self.data_gmd_raw)

        # country classifications, options
        self.list_countries = []
        self.country_file_dict = {}
        for skimname in range(len(self.list_raw)):
            cname = self.list_raw[skimname][:3]
            self.list_countries.append(cname)
            self.country_file_dict[cname] = self.list_raw[skimname]

        self.rich_countries = []#'AUT','BEL']
        self.countries_to_skip = []#['ARG','BGD','BFA','BEN','AFG','BLR','BOL','BDI']#'BGR'

        # drivers of scenarios
        self.uncertainties = ['shareag','sharemanu','shareemp','grserv','grag','grmanu','skillpserv','skillpag','skillpmanu','p','b','voice']
        self.lhssample     = pd.read_csv(self.scenar_folder+"lhs-table-600-12uncertainties.csv")
        self.ranges = pd.DataFrame(columns=['min','max'],index=self.uncertainties)
        self.impact_scenars = pd.read_csv(self.scenar_folder+"impact-scenarios-low-high.csv")

    def get_ini_year(self):

        df_ini_year = pd.DataFrame(columns={'ini_year':None},index=self.list_countries)

        for i in self.country_file_dict:

            df = pd.read_stata(self.data_gmd_raw+self.country_file_dict[i])
            df_ini_year.loc[i,'ini_year'] = df.iloc[0]['year']

        df_ini_year.to_csv(self.data_gmd+'/ini_year.csv')
        # for dta in range(len(self.list_raw)):
            # print(dta)

    def dta_to_csv(self):
        #list_countries = [_ for _ in os.listdir(data_gmd_dta) if 'DS_Store' not in _]
        for dta in range(len(self.list_raw)):
            dta_code = self.list_raw[dta][0]+self.list_raw[dta][1]+self.list_raw[dta][2]

            pd.read_stata(data_gmd_raw+list_raw[dta]).to_csv(self.data_gmd_raw+'/'+dta_code+'.csv')
            print('--> wrote {} to csv'.format(dta_code))


    def load_gmd_record(self,reset=False):
        try: 
            if reset: assert(False)
            record = pd.read_csv(self.data_gmd+'/GMD_to_finalhhframe_status.csv').set_index('country')
        except: 
            record = pd.DataFrame({'is_high_income':None,
                                   'GMD_has_sectoral_employment_data':None,
                                   'GMD_has_skill_level':None,
                                   'GMD_other_failures':None,
                                   'skim_is_final':None,
                                   'BAU_complete':None,
                                   'CC_complete':None},index=self.list_countries)
            record.index.name='country'

        record = record.loc[~record.index.duplicated(keep='first')]
        return record        


    def raw_to_skims(self,subset=None,reset=False):
        record = self.load_gmd_record(reset)

        if not subset: subset = self.list_countries 
        for countrycode in subset:

            if countrycode in self.countries_to_skip: continue # hard-coded above

            try: wbreg = self.codes.loc[self.codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values
            except: print('cant find wbreg for ',countrycode)

            # record if high income
            record.loc[countrycode,'is_high_income'] = True if wbreg == 'YHI' else False


            print('\n--> checking',countrycode)
            # Check if this has completed
            has_completed=False
            for cout in os.listdir(self.finalhhdataframes):
                if countrycode in cout: 
                    test_df = pd.read_csv(self.finalhhdataframes+cout)
                    if (test_df['cat1workers'].sum() != 0 
                        and test_df['cat2workers'].sum() != 0 
                        and test_df['cat3workers'].sum() != 0 
                        and test_df['cat4workers'].sum() != 0 
                        and test_df['cat5workers'].sum() != 0 
                        and test_df['cat6workers'].sum() != 0 
                        and test_df['cat7workers'].sum() != 0): has_completed = True

            # if countrycode in record.index and record.loc[countrycode,'skim_is_final']==True: continue
    		# if countrycode in record.index and not record.loc[countrycode,'GMD_has_sectoral_employment_data']: continue
    		# if countrycode in record.index and not record.loc[countrycode,'GMD_has_skill_level']: continue
            if has_completed: continue
            print('\n--> running',countrycode)

                
            try:
            # if True:
                finalhhframe, failure_types = create_correct_data(self,countrycode)
                has_sectoral_employment_data,has_skill_level = failure_types
                record.loc[countrycode,'GMD_has_sectoral_employment_data'] = has_sectoral_employment_data
                record.loc[countrycode,'GMD_has_skill_level'] = has_skill_level
                
                if not has_sectoral_employment_data or not has_skill_level:
                    print('did not save out final df for '+countrycode)
                    record.loc[countrycode,'skim_is_final'] = False
                    
                else:
                    if finalhhframe['idh'].dtype=='O':
                        try:
                            finalhhframe['idh']=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
                            finalhhframe['idh']=finalhhframe['idh'].astype(float)
                        except: pass
                    
                    finalhhframe = finalhhframe.dropna(how='all',axis=1)
                    finalhhframe.to_csv(self.finalhhdataframes+countrycode+'_finalhhframe.csv',encoding='utf-8',index=False)
                    print('got final df for '+countrycode)
                    record.loc[countrycode,['GMD_other_failures','skim_is_final']] = [False,True]
                
            except:
                print('...multiple failures')
                record.loc[countrycode,['GMD_other_failures','skim_is_final']] = [True,False]
            
            # update GMD record
            record.sort_index().to_csv(self.data_gmd+'GMD_to_finalhhframe_status.csv')
