from pandas import Series,DataFrame,merge,read_excel,read_csv
import numpy as np
import cvxopt
from cvxopt.solvers import qp
from perc import *
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from scipy import integrate, optimize
from kde import gaussian_kde
from lib_for_data_reading import *
from lib_for_cc import *
import scipy

from statsmodels.nonparametric.kde import kdensity

################# Functions used to define the scenarios (ranges are specific to each country) #########################################

def scenar_ranges(ssp,ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year,b_value,verbose=True):
	'''
	This is a messy function at the moment. It sets the ranges of uncertainties. 
	For redistribution (p and b) it is just a fixed range. 
	For structural change, the ranges depend on the initial shares and are calculated in find_range_struct. 
	For growth rates,xxx 
	'''
	
	characteristics = keep_characteristics_to_reweight(finalhhframe)
	ini_pop_desc    = calc_pop_desc(characteristics,finalhhframe['weight'])
	if verbose: print('Initial pop description:\n',ini_pop_desc)
	
	shareemp_ini,shareag_ini,sharemanu_ini,share_skilled, shareurban_ini = indicators_from_pop_desc(ini_pop_desc)
	if verbose: print('share employed_i:',shareemp_ini,
			  '\nshare ag_i:',shareag_ini,
			  '\nshare manu_i:',sharemanu_ini,
			  '\nshare skilled_i:',share_skilled,
			  '\nshare urban_i:',shareurban_ini)
	
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	urban         = float(ini_pop_desc['urban'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])
	
	ini_pop_out = ini_pop_desc.copy()
	ini_pop_out['employed'] = work
	ini_pop_out['adults'] = adults
	#
	for _ in ['1','2','3','4','5','6']:
		ini_pop_out['cat'+_+'workers'] = finalhhframe[['cat'+_+'workers','weight']].prod(axis=1).sum()
		#
		ini_pop_out['cat'+_+'workers_urban'] = finalhhframe.loc[finalhhframe.urban!=0,['cat'+_+'workers','weight']].prod(axis=1).sum()
		ini_pop_out['cat'+_+'workers_rural'] = finalhhframe.loc[finalhhframe.urban==0,['cat'+_+'workers','weight']].prod(axis=1).sum()
	#


	#ssp_growth = np.mean([gr3])
	#if verbose: print('\n\nSSP/macro GDP growth params:',gr3,ssp_growth)

	gr=(get_gdp_growth(ssp_gdp,year,ssp,country2r32(codes_tables,countrycode),ini_year))**(1/(year-ini_year))-1
	ssp_growth = np.mean([gr])
	
	pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,ssp,year,countrycode)
	pop_growth = np.mean([(pop_1564/adults)**(1/(year-ini_year))-1])
	#pop_tot,pop_0014,pop_1564_4,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,4,year,countrycode)
	#pop_tot,pop_0014,pop_1564_5,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,5,year,countrycode)
	#pop_growth = np.mean([(pop_1564_5/adults)**(1/(year-ini_year))-1,(pop_1564_5/adults)**(1/(year-ini_year))-1])
	if verbose: print('SSP/population growth ('+str(year)+')',pop_tot,pop_growth)

	
	ranges.ix['shareag',['min','max']]    = find_range_struct(shareag_ini,'ag')
	ranges.ix['sharemanu',['min','max']]  = find_range_struct(sharemanu_ini,'ind')
	ranges.ix['shareemp',['min','max']]   = find_range_struct(shareemp_ini,'emp')
	ranges.ix['shareurban',['min','max']] = load_ssp_urbanization_data(countrycode,shareurban_ini,ini_year,year,ssp=2)
	
	select_gr=['grag','grmanu','grserv']
	ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.05,ssp_growth-pop_growth+0.01]
    
	# The code below is unnecessary and needs to be updated
	#if countrycode in ['BTN']:
		#ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.08,ssp_growth-pop_growth-0.01]
	#if countrycode in ['AFG','ALB','BIH','CHN','DOM','ECU','EGY','FSM','GEO','GIN','JAM','KGZ','MAR','MDA','MKD','MNG','MOZ','NPL','PHL']:
		#ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.06,ssp_growth-pop_growth]
	#if countrycode in ['BDI']:
		#ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.04,ssp_growth-pop_growth+0.03]
	#if countrycode in ['TCD','ZMB','SWZ']:
		#ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.02,ssp_growth-pop_growth+0.04]
		
	
	skillp_file = read_csv('skill_premiums.csv').set_index('countrycode')
	ranges.ix[['skillpserv'],['min','max']] = [max(1.0*float(skillp_file.ix[countrycode,'service']),1.),1.0*float(skillp_file.ix[countrycode,'service'])]
	ranges.ix[['skillpag'],  ['min','max']] = [max(1.0*float(skillp_file.ix[countrycode,'agricul']),1.),1.0*float(skillp_file.ix[countrycode,'agricul'])]
	ranges.ix[['skillpmanu'],['min','max']] = [max(1.0*float(skillp_file.ix[countrycode,'manufac']),1,),1.0*float(skillp_file.ix[countrycode,'manufac'])]

	ranges.ix['p',['min','max']]=[0.001,0.2]
	ranges.ix['b',['min','max']]=[b_value,b_value]
	
	ranges.ix['voice',['min','max']]=[0,1]

	if verbose: print('\n\nFit parameter inputs (ranges):\n',ranges.head(15))
	return ranges, ini_pop_out
	
	


def load_ssp_urbanization_data(countrycode,shareurban_ini,ini_year,year,ssp=2):
	_ssp = read_excel('ssp_data/urbproj_all.xlsx',sheetname='data')
	try: 
		result = 1E-2*float(_ssp.loc[(_ssp['Region']==countrycode)&(_ssp['Scenario']=='SSP{}_v9_131230'.format(int(ssp))),year])
		if result < shareurban_ini: 
			print('\n\nFuture urbanization (=',result,'from SSP) > current (=',shareurban_ini,'from survey). Could create issue...')
		return max(result,shareurban_ini)
	except: return min(shareurban_ini,1.0)




def find_range_struct(ini_share,sector):
	ini_share = float(ini_share)
	
	x    = [0,0.01,0.1,0.3,0.5,0.7,0.9,1]
	if sector=='ag':
		ymin = [0,0.001,0.01,0.1,0.2,0.3,0.4,0.6]
		ymax = [0,0.2,0.15,0.4,0.5,0.6,0.8,0.8]
	elif sector=='ind':
		ymin = [0,0.1,0.15,0.1,0.2,0.3,0.35,0.4]
		ymax = [0,0.25,0.3,0.35,0.4,0.5,0.7,0.8]
	elif sector=='emp':
		ymin = [0,0.007,0.07,0.25,0.4,0.6,0.75,0.8]
		ymax = [0,0.4,0.5,0.6,0.7,0.9,0.99,1]
	elif sector=='urban':
		ymin = [0,0.007,0.07,0.25,0.4,0.6,0.75,0.8]
		ymax = [0,0.4,0.5,0.6,0.7,0.9,0.99,1]
		# lifted these values from 'emp' -- talk to Julie about what they represent  

	w         = [1,2,2,2,1,1,1,1]
	smin      = UnivariateSpline(x, ymin, w)
	smax      = UnivariateSpline(x, ymax, w)
	range_out = [max(float(smin(ini_share)),0),min(float(smax(ini_share)),1)]
	return range_out
	
def correct_shares(shareag,sharemanu):
	if shareag+sharemanu>1:
		tot=shareag+sharemanu
		shareag=shareag/tot-0.001
		sharemanu=sharemanu/tot-0.001
	return shareag,sharemanu
	


####### Functions used for the reweighting process #######################################

def calc_pop_desc(characteristics,weights):
	pop_description             = DataFrame(columns=characteristics.columns)
	pop_description.ix['pop',:] = np.dot(characteristics.T,weights)
	return pop_description

	
def keep_characteristics_to_reweight(finalhhframe):
	characteristics                 = DataFrame()
	characteristics['old']          = finalhhframe['old']
	characteristics['children']     = finalhhframe['children']
	characteristics['unemployed']   = finalhhframe['cat7workers']
	characteristics['skillworkers'] = finalhhframe['cat2workers']+finalhhframe['cat4workers']+finalhhframe['cat6workers']
	characteristics['servworkers']  = finalhhframe['cat1workers']+finalhhframe['cat2workers']
	characteristics['agworkers']    = finalhhframe['cat3workers']+finalhhframe['cat4workers']
	characteristics['manuworkers']  = finalhhframe['cat5workers']+finalhhframe['cat6workers']
	characteristics['urban']        = finalhhframe['urban'] 
	return characteristics
	
def build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,shareurban,istransformationscen,ischildren=False):
	'''builds a new description vector for the projected year, from ssp data and exogenous share for skilled people and agri people'''
	pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,ssp,year,countrycode)
	pop_tot_GLO, pop_0014_GLO, pop_1564_GLO,pop_65up_GLO,skilled_adults_GLO = get_pop_data_from_ssp(ssp_pop,ssp,year,countrycode,_global=True)
		
	new_pop_desc                 = ini_pop_desc.copy()

	new_pop_desc['children']     = ini_pop_desc['children']
	if ischildren: 
		new_pop_desc['children'] = pop_0014

	new_pop_desc['old']          = pop_65up
	new_pop_desc['skillworkers'] = skilled_adults*shareemp
	new_pop_desc['agworkers']    = pop_1564*shareag*shareemp
	new_pop_desc['manuworkers']  = pop_1564*sharemanu*shareemp
	new_pop_desc['servworkers']  = pop_1564*(1-shareag-sharemanu)*shareemp
	new_pop_desc['unemployed']   = pop_1564*(1-shareemp)
	new_pop_desc['urban']        = pop_tot*shareurban

	if not istransformationscen: 
		return new_pop_desc,pop_0014
		
	tmp_pop_desc = new_pop_desc.copy()  
	
	# FRUITS & NUTS
	# Shift 5% of unemployed into agriculture
	unemp_to_ag = 0.05*tmp_pop_desc['unemployed']
	new_pop_desc['unemployed']   -= unemp_to_ag
	new_pop_desc['agworkers']    += unemp_to_ag
	
	# 30M into high skilled farming (Global total)
	unemp_to_ag = 30E6*(pop_1564/pop_1564_GLO)
	new_pop_desc['unemployed']   -= unemp_to_ag
	new_pop_desc['agworkers']    += unemp_to_ag
	new_pop_desc['skillworkers'] += unemp_to_ag

	# 30M into low skilled farming (Global total)
	unemp_to_ag = 30E6*(pop_1564/pop_1564_GLO)
	new_pop_desc['unemployed']   -= unemp_to_ag
	new_pop_desc['agworkers']    += unemp_to_ag

	# 45M increase in high-skill sv
	unemp_to_sv = 45E6*(pop_1564/pop_1564_GLO)
	new_pop_desc['unemployed'] -= unemp_to_sv
	new_pop_desc['servworkers'] += unemp_to_sv 
	new_pop_desc['skillworkers'] += unemp_to_sv

	# 60M increase in low-skill sv
	unemp_to_sv = 45E6*(pop_1564/pop_1564_GLO)
	new_pop_desc['unemployed'] -= unemp_to_sv
	new_pop_desc['servworkers'] += unemp_to_sv

	# 60M increase from low-skill ag to low-skill mn
	ag_to_mn = 60E6*(pop_1564/pop_1564_GLO)
	new_pop_desc['agworkers'] -= ag_to_mn
	new_pop_desc['manuworkers'] += ag_to_mn

	return new_pop_desc,pop_0014


	
def futurehh(finalhhframe,pop_0014,ischildren=False):
	'''
	ischildren is True if the number of children is taken into account in the re-weighting process. Otherwise, we re-scale the number of people based on the new number of children.
	'''
	futurehhframe=finalhhframe.copy()
	if not ischildren:
		futurehhframe['children']=finalhhframe['children']*pop_0014/sum(finalhhframe['children']*finalhhframe['weight'])
	futurehhframe['nbpeople']=finalhhframe['nbpeople']+futurehhframe['children']-finalhhframe['children']
	futurehhframe.drop(['Y','weight'], axis=1, inplace=True)
	return futurehhframe
	
def build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True):
	'''optimize new weights to match current households and new population description'''


	t_tilde = cvxopt.matrix((future_pop_desc.values-ini_pop_desc.values).astype(np.float,copy=False))
	aa      = cvxopt.matrix(characteristics.values.astype(np.float,copy=False))
	w1      = 1/(ini_weights.values)**2
	n       = len(w1)
	P       = cvxopt.spdiag(cvxopt.matrix(w1))
	G       = -cvxopt.matrix(np.identity(n))
	h       = cvxopt.matrix(ini_weights.values.astype(np.float,copy=False))
	q       = cvxopt.matrix(0.0,(n,1))

	if ismosek:
		result = qp(P,q,G,h,aa.T,t_tilde.T,solver='mosek')['x']
	else:
		result = qp(P,q,G,h,aa.T,t_tilde.T)['x']
	if result is None:
		new_weights = 0*ini_weights
	else:
		new_weights = ini_weights+list(result)
	return new_weights
	
def find_new_weights(characteristics,ini_weights,future_pop_desc):
	ini_pop_desc = calc_pop_desc(characteristics,ini_weights)
	weights_proj = build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True)
	weights_proj = Series(np.array(weights_proj),index=ini_weights.index.values,dtype='float64')
	return weights_proj
	




##############################################################################
########### Functions used before changing future income #####################
##############################################################################

ind_dict = read_csv('scenar_def/list_industries.csv').set_index('indata').to_dict()['industrycode']
def apply_industry_dict(survey_response,ind_dict=ind_dict):
	#print(ind_dict)
	try: survey_response = str(int(survey_response))
	except: pass

	if survey_response in ind_dict:
		try: return ind_dict[survey_response]
		except: return ind_dict[str(survey_response)]

	#print(survey_response,type(survey_response))
	return None

def estime_income(hhcat,finalhhframe,countrycode,year,with_person_data=True):
	if with_person_data:

		gmd_skim = read_csv('GMD_skims/'+countrycode+'.csv').set_index('hhid')

		if 'datalevel' not in gmd_skim.columns:
			if countrycode == 'CHN': 
				gmd_skim['datalevel'] = 0
				gmd_skim.loc[gmd_skim['urban']==1,'datalevel']=1

		_income = 'labor_ind' if 'labor_ind' in gmd_skim.columns else 'welfare'
		_weight = 'weight_p' if 'weight_p' in gmd_skim.columns else 'weight'
		#print(_weight)

		if (countrycode != 'IND' and countrycode != 'IDN' and countrycode != 'CHN'): gmd_skim['datalevel'] = 2

		conv_df = read_stata('finalhhdataframes_GMD/_Final_CPI_PPP_to_be_used.dta')[['code','year','datalevel','cpi2011','icp2011']]
		conv_df = conv_df.loc[(conv_df.code==countrycode)&(conv_df.year==year)]

		gmd_skim['ppp'] = None
		if (countrycode == 'IND' or countrycode == 'IDN' or countrycode == 'CHN'):
			gmd_skim.loc[gmd_skim['datalevel']==0,'ppp'] = 1/float(conv_df.loc[conv_df['datalevel']==0,['cpi2011','icp2011']].prod(axis=1))
			gmd_skim.loc[gmd_skim['datalevel']==1,'ppp'] = 1/float(conv_df.loc[conv_df['datalevel']==1,['cpi2011','icp2011']].prod(axis=1))
		else: gmd_skim['ppp'] = 1/float(conv_df.loc[conv_df['datalevel']==2,['cpi2011','icp2011']].prod(axis=1))

		gmd_skim[_income]*= gmd_skim['ppp']

		gmd_skim['industry_cat'] = None
		gmd_skim['industry_cat'] = gmd_skim['industrycat10'].apply(lambda x:apply_industry_dict(x,ind_dict)).astype('str')

		if 'educat4' in gmd_skim.columns:
			skilled = gmd_skim.loc[gmd_skim.educat4>=3,[_weight,_income]].prod(axis=1).sum()/gmd_skim.loc[gmd_skim.educat4>=3,_weight].sum()
			unskilled = gmd_skim.loc[gmd_skim.educat4<3,[_weight,_income]].prod(axis=1).sum()/gmd_skim.loc[gmd_skim.educat4<3,_weight].sum()
		elif gmd_skim['educy'].sum() > 0:
			skilled = gmd_skim.loc[gmd_skim.educy>9,[_weight,_income]].prod(axis=1).sum()/gmd_skim.loc[gmd_skim.educy>9,'weight'].sum()
			unskilled = gmd_skim.loc[gmd_skim.educy<=9,[_weight,_income]].prod(axis=1).sum()/gmd_skim.loc[gmd_skim.educy<=9,'weight'].sum()


		# use this function to get the average income in the following categories:
		# (ag,service,manu)X(skilled,unskilled)

		out = Series({'avg_income_pc':None},index={})
		_catcode = 1
		for _ind in ['serv','ag','manu']:

			if 'educat4' in gmd_skim.columns:
				out['cat'+str(_catcode)+'workers'] = (gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educat4']<3),[_weight,_income]].prod(axis=1).sum()
								      /gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educat4']<3),_weight].sum())
				out['cat'+str(_catcode+1)+'workers'] = (gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educat4']>=3),[_weight,_income]].prod(axis=1).sum()
									/gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educat4']>=3),_weight].sum())
			
			elif gmd_skim['educy'].sum() > 0:
				out['cat'+str(_catcode)+'workers'] = (gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educy']<=9),[_weight,_income]].prod(axis=1).sum()
								      /gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educy']<=9),_weight].sum())
				out['cat'+str(_catcode+1)+'workers'] = (gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educy']>9),[_weight,_income]].prod(axis=1).sum()
									/gmd_skim.loc[(gmd_skim['industry_cat']==_ind)&(gmd_skim['educy']>9),_weight].sum())
			_catcode+=2
		
		# calculate skill premium
		_skillprem_file = read_csv('skill_premiums.csv').set_index('countrycode')
		_skillprem_file = _skillprem_file.append(DataFrame({'service':float(out['cat2workers']/out['cat1workers']),
								    'agricul':float(out['cat4workers']/out['cat3workers']),
								    'manufac':float(out['cat6workers']/out['cat5workers'])},index=[countrycode]))

		_skillprem_file.index.name = 'countrycode'
		_skillprem_file.loc[~(_skillprem_file.index.duplicated(keep='last'))].to_csv('skill_premiums.csv')

		out['cat7workers'] = (gmd_skim.loc[(gmd_skim['industry_cat']=='None'),[_weight,_income]].prod(axis=1).sum()
				      /gmd_skim.loc[(gmd_skim['industry_cat']=='None'),_weight].sum())		
		
		out['old'] = (gmd_skim.loc[(gmd_skim['age']>64),[_weight,_income]].prod(axis=1).sum()
			      /gmd_skim.loc[(gmd_skim['age']>64),_weight].sum())

		return out
	else:
		select     = finalhhframe.Y<float(perc_with_spline(finalhhframe.Y,finalhhframe.weight*finalhhframe.nbpeople,0.95))
		X          = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
		w          = finalhhframe.ix[select,'weight'].copy()
		w[w==0]    = 10**(-10)
		Y          = (finalhhframe.ix[select,'Y']*finalhhframe.ix[select,'nbpeople'])
		result     = sm.WLS(Y, X, weights=1/w).fit()
		inc        = result.params
		nonworkers = inc[['cat7workers','old']].copy()
		negs       = nonworkers[nonworkers<0].index
		if len(negs)>0:
			X.drop(negs.values,axis=1,inplace=True)
			result = sm.WLS(Y, X, weights=1/w).fit()
			inc    = result.params
			for ii in negs:
				inc[ii] = 0
		a        = result.pvalues
		nonsign1 = a[a>0.05].index
		nonsign2 = []
		nonsign3 = []
		if len(nonsign1)>0:
			X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
			X['serv'] = X['cat1workers']+X['cat2workers']
			X['ag']   = X['cat3workers']+X['cat4workers']
			X['manu'] = X['cat5workers']+X['cat6workers']
			X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
			result3   = sm.WLS(Y, X, weights=1/w).fit()
			a3        = result3.pvalues
			nonsign3  = a3[a3>0.05].index
			if (len(nonsign3)==0):
				inctemp            = result3.params
				inc['cat2workers'] = inctemp['serv']
				inc['cat4workers'] = inctemp['ag']
				inc['cat6workers'] = inctemp['manu']
				inc['cat1workers'] = inctemp['serv']
				inc['cat3workers'] = inctemp['ag']
				inc['cat5workers'] = inctemp['manu']
			else:
				X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
				X['skilled']   = X['cat2workers']+X['cat4workers']+X['cat6workers']
				X['unskilled'] = X['cat1workers']+X['cat3workers']+X['cat5workers']
				X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
				result2        = sm.WLS(Y, X, weights=1/w).fit()
				a2             = result2.pvalues
				nonsign2       = a2[a2>0.05].index
				if len(nonsign2)==0|((len(nonsign2)<len(nonsign1))&(len(nonsign2)<len(nonsign3))):
					inctemp           = result2.params
					inc['cat2workers']= inctemp['skilled']
					inc['cat4workers']= inctemp['skilled']
					inc['cat6workers']= inctemp['skilled']
					inc['cat1workers']= inctemp['unskilled']
					inc['cat3workers']= inctemp['unskilled']
					inc['cat5workers']= inctemp['unskilled']
				else:
					if (len(nonsign3)<len(nonsign1))&(len(nonsign3)<len(nonsign2)):
						inctemp            = result3.params
						inc['cat2workers'] = inctemp['serv']
						inc['cat4workers'] = inctemp['ag']
						inc['cat6workers'] = inctemp['manu']
						inc['cat1workers'] = inctemp['serv']
						inc['cat3workers'] = inctemp['ag']
						inc['cat5workers'] = inctemp['manu']
		return inc

#
#moved below function to lib_for_country_run()
# estime_income is in this script (above)
#
#def estimate_income_and_all(hhcat,hhdataframe):
#	inc             = estime_income(hhcat,hhdataframe)
#	characteristics = keep_characteristics_to_reweight(hhdataframe)
#	ini_pop_desc    = calc_pop_desc(characteristics,hhdataframe['weight'])
#	inimin          = hhdataframe['Y'].min()
#	return inc,characteristics,ini_pop_desc,inimin
	
############ Functions used for changing household income ##############################
	
def Y_from_inc(futurehhframe,inc):
	'''
	Calculates total hh income as the sum of each people's income in the household (estimated income)
	'''
	listofvariables = list(inc.index)
	out             = 0*futurehhframe['nbpeople']
	for var in listofvariables:
		out += inc[var]*futurehhframe[var]
	Ycalc           = Series(out,index=out.index)
	return Ycalc
	
def before_tax(inc,finalhhframe):
	'''Calculates the pre-tax revenues, assuming that elderly and unemployed incomes come from redistribution only. The error term (difference btw calculated and actual income) is included in the taxed revenue. We therefore calculate a pre-tax error term.
	Note: obsolete in the latest version of the model.
	'''
	inc_bf=inc.copy()
	errorterm=finalhhframe['totY']-Y_from_inc(finalhhframe,inc)
	gdpobserved=GDP(finalhhframe['totY'],finalhhframe['weight'])
	pensions=GDP(Y_from_inc(finalhhframe,inc[['old']]),finalhhframe['weight'])
	benefits=GDP(Y_from_inc(finalhhframe,inc[['cat7workers']]),finalhhframe['weight'])
	p=pensions/gdpobserved
	b=benefits/gdpobserved
	for thecat in range(1,7):
		string='cat{}workers'.format(int(thecat))
		inc_bf[string]=inc[string]*1/(1-p-b)
	inc_bf['old']=0
	inc_bf['cat7workers']=0
	errorterm=errorterm*1/(1-p-b)
	return b,p,errorterm,inc_bf
	
def keep_workers(inputs):
	thebool=(inputs.index!='old')&(inputs.index!='cat7workers')
	return thebool

	
def after_pensions(inc,errorterm,p,finalhhframe):
	'''
	Transfers income from workers to retirees. The error term is taxed also, only for households that are not only composed of unemployed or elderlies.
	'''
	inigdp = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	select = ~((finalhhframe['cat7workers']+finalhhframe['old'])==(finalhhframe['nbpeople']-finalhhframe['children']))
	inc_af = inc.copy()
	for thecat in range(1,7):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-p)
	errorterm[select] = errorterm[select]*(1-p)
	totrev            = inigdp-GDP(Y_from_inc(finalhhframe,inc_af)+errorterm,finalhhframe['weight'])
	pensions          = totrev/sum(finalhhframe['old']*finalhhframe['weight'])
	inc_af['old']     = inc['old']+pensions
	return inc_af,errorterm
	
def after_bi(countrycode,inc,errorterm,b,finalhhframe,desc_str):
	'''Recalculates the after-basic-income incomes. All categories are taxed (including unemployed and retirees) and all adults receive the basic income. The error term is taxed also.'''
	str_b  = str(int(round(1E2*b)))
	inc_af  = inc.copy()
	gdpcalc = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	bI      = b*gdpcalc/sum((finalhhframe['nbpeople']-finalhhframe['children'])*finalhhframe['weight'])

	#open file
	try: basic_income_stats_out = read_csv('basic_income_stats_out.csv').set_index('country')
	except: 
		basic_income_stats_out = DataFrame(columns={('total cost mil. (b={}% ssp{})').format(str_b,desc_str):None,
							    ('income per adult (b={}% ssp{})').format(str_b,desc_str):None},index={countrycode})
		basic_income_stats_out.index.name = 'country'
		
	basic_income_stats_out.loc[countrycode,'total cost mil. (b={}% ssp{})'.format(str_b,desc_str)] = 1E-6*b*gdpcalc
	basic_income_stats_out.loc[countrycode,'income per adult (b={}% ssp{})'.format(str_b,desc_str)] = bI
	basic_income_stats_out = basic_income_stats_out.loc[~(basic_income_stats_out.index.duplicated(keep='last'))]
	basic_income_stats_out.to_csv('basic_income_stats_out.csv')
	basic_income_stats_out.stack().to_csv('basic_income_stats_out_stacked.csv')

	for thecat in range(1,8):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-b)+bI
	inc_af['old'] = inc['old']*(1-b)+bI
	errorterm     = errorterm*(1-b)
	return inc_af,errorterm
		

def future_income_simple(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,price_increase,desc_str):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''
	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	
	b = inputs['b']

	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)*(1+price_increase)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	futureinc = temperature_impact(futureinc,shares_outside,temp_impact)

	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(countrycode,futureinc,futurerrorterm,b,futurehhframe,desc_str)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc

def get_frac_forest(countrycode):
	df = read_excel('../tropical_forests.xlsx',sheetname='Sheet2').set_index('countrycode')
	if countrycode in df.index:
		return float(df.loc[countrycode,'sq_km']/df['sq_km'].sum())
	return 0.

def future_income_transformation(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,price_increase,desc_str):
	#
	do_ecoservices_payments = True
	#
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained (non-wage) income comes from.
	'''

	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']
	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)*(1+price_increase)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']

	futureinc = temperature_impact(futureinc,shares_outside,temp_impact)

	# Middle-skilled workers
	tmp_df = futurehhframe[['cat3workers','totY','weight']].copy().sort_values('totY',ascending=False)
	tmp_df['frac'] = tmp_df[['weight','cat3workers']].prod(axis=1).cumsum()/float(tmp_df[['weight','cat3workers']].prod(axis=1).sum())
	tmp_df['middle_skilled'] = 0
	tmp_df.loc[tmp_df.frac<=0.30,'middle_skilled'] = 1
	tmp_df = tmp_df.sort_index()

	futurehhframe.loc[tmp_df['middle_skilled']==1,'cat3workers'] *= (1 + 0.5*(inputs['skillpag']-1))

	# Ecosystems services payments
	global_ecoservices_benefit = 30E9
	futureinc['cat3workers'] += global_ecoservices_benefit*get_frac_forest(countrycode)/finalhhframe[['cat3workers','weight']].prod(axis=1).sum()

	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(countrycode,futureinc,futurerrorterm,b,futurehhframe,desc_str)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc
	
def future_income_baseline(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,desc_str,future_with_cc=False):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''

	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']
	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(countrycode,futureinc,futurerrorterm,b,futurehhframe,desc_str)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc



###################### In progress: functions to split households with a very high weight into several types of households ###########

def weighted_std(values, weights):
	'''Weighted standard deviation'''
	average = np.average(values, weights=weights)
	variance = np.average((values-average)**2, weights=weights)
	return np.sqrt(variance)

def create_new_hh(a_serie,chosen_std):
	'''
	Creates new households based on those that have very high weights after the re-weighting process (currently not used but in progress)
	'''

	#h=med/m = exp(-s**2/2)   log(h) = -s**2/2
	#h=med/m   med = m*h = exp(mu)   mu = log(m*h)
	h = np.exp(-chosen_std**2/2)
	norm=scipy.stats.lognorm(s=chosen_std,loc=np.log(h*a_serie.Y))
	x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 50)
	oo = DataFrame(columns=a_serie.index)
	oo = oo.append([a_serie]*50,ignore_index=True)
	oo['Y'] = x
	oo.ix[oo['Y']<0,'Y']=0
	oo['weight'] = oo['weight']*norm.pdf(x)/sum(norm.pdf(x))
	oo.index = [str(a_serie.name)+"_"+str(i) for i in oo.index]
	return oo
	
def add_errors_to_distrib(futurehhframe_old,e,w_th):
	'''
	Creates normal income distributions around households that have very high weights after the re-weighting process (currently not used but in progress)
	'''
	futurehhframe = futurehhframe_old.copy()
	cat_df = futurehhframe.ix[futurehhframe.Y<float(perc_with_spline(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,0.9)),:].groupby(['cat1workers', 'cat2workers','cat3workers', 'cat4workers', 'cat5workers', 'cat6workers','cat7workers']).apply(lambda x:weighted_std(x.Y,x.weight*x.nbpeople))
	chosen_std = np.median(cat_df[cat_df>0])/e
	select = futurehhframe.weight*futurehhframe.nbpeople>w_th*sum(futurehhframe.weight*futurehhframe.nbpeople)
	t = futurehhframe.ix[select,:].copy()
	futurehhframe = futurehhframe.drop(futurehhframe.ix[select,:].index)
	new_households = DataFrame()
	for index, row in t.iterrows():
		new_households = new_households.append(create_new_hh(row,chosen_std))
	futurehhframe = futurehhframe.append(new_households)
	return futurehhframe


###################### Functions used to calculate indicators at the end of the run ############################################

def GDP(income,weights):
	GDP = np.nansum(income*weights)
	return GDP
	
def actual_productivity_growth(finalhhframe,inc,futurehhframe,futureinc,year,ini_year):
	
	out=list()
	
	for cat1,cat2 in (['cat1workers','cat2workers'],['cat3workers','cat4workers'],['cat5workers','cat6workers']):
		prod_ini  = (sum(finalhhframe[cat1]*finalhhframe['weight'])*inc[cat1]+sum(finalhhframe[cat2]*finalhhframe['weight'])*inc[cat2])/sum((finalhhframe[cat1]+finalhhframe[cat2])*finalhhframe['weight'])
		prod_last = (sum(futurehhframe[cat1]*futurehhframe['weight'])*futureinc[cat1]+sum(futurehhframe[cat2]*futurehhframe['weight'])*futureinc[cat2])/sum((futurehhframe[cat1]+futurehhframe[cat2])*futurehhframe['weight'])
		
		prod_gr   = (prod_last/prod_ini)**(1/(year-ini_year))-1
		out.append(prod_gr)
	return tuple(out)
	
def poor_people(income,weights,povline):
	isbelowline = (income<povline)
	thepoor     = weights.values*isbelowline.values
	nbpoor      = thepoor.sum()
	return nbpoor
	
def find_perc(y,w,theperc,density):
	'''
	The very sophisticated way of finding percentiles
	'''
	normalization = integrate.quad(density,0,np.inf)
	estime = wp(y,w,[theperc],cum=False)
	def find_root(x,normalization,density,theperc):
		integrale = integrate.quad(density,0,x)
		return integrale[0]/normalization[0]-theperc
	out = optimize.fsolve(find_root, estime, args=(normalization,density,theperc))
	return out
	
def poverty_indic_kde(income,weights,threshold,density):
	'''
	The very sophisticated way of finding percentiles and the average income of people between percentiles
	'''
	if type(threshold)==float:
		inclim20        = find_perc(income,weights,threshold,density)[0]
		isbelowline     = (income<=inclim20)
	elif type(threshold)==list:
		minlim = find_perc(income,weights,threshold[0],density)[0]
		maxlim = find_perc(income,weights,threshold[1],density)[0]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic_spec(income,weights,threshold):
	'''
	For special cases
	'''
	if type(threshold)==list:
		minlim = threshold[0]
		maxlim = threshold[1]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	else:
		isbelowline     = (income<=threshold)
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic(percentiles,limit1,limit2):
	out = percentiles[limit1:limit2].sum()/(limit2-limit1)
	return out
	
def gini(income,weights):
	inc = np.asarray(reshape_data(income))
	wt  = np.asarray(reshape_data(weights))
	i   = np.argsort(inc) 
	inc = inc[i]
	wt  = wt[i]
	y   = np.cumsum(np.multiply(inc,wt))
	y   = y/y[-1]
	x   = np.cumsum(wt)
	x   = x/x[-1]
	G   = 1-sum((y[1::]+y[0:-1])*(x[1::]-x[0:-1]))
	return G
	
def poverty_gap(income,weights,povline):
	isbelowline = (income<povline)
	gap         = sum((1-income[isbelowline]/povline)*weights[isbelowline]/sum(weights))
	return gap
		
def distrib2store(income,weights,nbdots,tot_pop):
	categories         = np.arange(0, 1+1/nbdots, 1/nbdots)
	y                  = np.asarray(wp(reshape_data(income),reshape_data(weights),categories,cum=False))
	inc_o              = (y[1::]+y[0:-1])/2
	o2store            = DataFrame(columns=['income','weights'])
	o2store['income']  = list(inc_o)
	o2store['weights'] = list([tot_pop/nbdots]*len(inc_o))
	return o2store
	
def indicators_from_pop_desc(ini_pop_desc):
	children      = ini_pop_desc['children']
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	urban         = float(ini_pop_desc['urban'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])
	earn_income   = adults+float(ini_pop_desc['old'])
	tot_pop       = earn_income+children
	skilled       = float(ini_pop_desc['skillworkers'])
	
	shareurban_ini = float(urban/tot_pop)
	shareemp_ini   = float(1-ini_pop_desc['unemployed']/adults)
	shareag_ini    = ag/work
	sharemanu_ini  = manu/work
	
	share_skilled = skilled/adults
	
	return shareemp_ini,shareag_ini,sharemanu_ini,share_skilled, shareurban_ini
	
		
def get_fa(countrycode):
	wb_income_class = read_csv('RedX_summary/wbccodes2014.csv',index_col='country')[['wbincomename']]
	#
	wb_income_class.loc['TLS'] = 'Low income'
	wb_income_class.loc['COD'] = 'Low income'

	ifrc_fa = read_excel('RedX_summary/Population_disaster_affect_rates_v2.xlsx',sheet_name='Sheet2')

	_out = float(ifrc_fa.loc[(ifrc_fa['income_class']==str((wb_income_class.loc[countrycode,'wbincomename'])
							       .replace('High income: OECD','High income')
							       .replace('High income: nonOECD','High income')))&(ifrc_fa['hazard']=='Overall'),'Mean.min'])

	return _out

def calc_indic(countrycode,income_proj,weights_proj_tot,weights_proj,futurehhframe,data2day,futureinc,povline,ini_year=None):
	'''
	Indicators that are calculated at the end and stored. Since we cannot store the entire survey for each scenario we summarize the information with these indicators.
	'''
	
	# Define slices
	sv = (futurehhframe['cat1workers']>0)|(futurehhframe['cat2workers']>0)
	sv_u = (futurehhframe['cat1workers']>0)
	sv_s = (futurehhframe['cat2workers']>0)
	ag = (futurehhframe['cat3workers']>0)|(futurehhframe['cat4workers']>0)
	ag_u = (futurehhframe['cat3workers']>0)
	ag_s = (futurehhframe['cat4workers']>0)
	mn = (futurehhframe['cat5workers']>0)|(futurehhframe['cat6workers']>0)
	mn_u = (futurehhframe['cat5workers']>0)
	mn_s = (futurehhframe['cat6workers']>0)
	#
	urb = (futurehhframe['urban']>0)
	rur = (futurehhframe['urban']==0)
	futurehhframe['total_workers'] = futurehhframe[['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers']].sum(axis=1)
	working = (futurehhframe['total_workers']>0)
	#

	# density = gaussian_kde(income_proj,weights=weights_proj_tot)
	# density._compute_covariance()
	
	indicators                 = DataFrame()
	#
	percentiles                = perc_with_spline(income_proj,weights_proj_tot,np.arange(0,1,0.01))
	indicators['avg_income_bott20'] = [poverty_indic(percentiles,0,20)]
	indicators['avg_income'] = [np.average(income_proj,weights=weights_proj_tot)]
	indicators['avg_income_ifrc_fa'] = [poverty_indic(percentiles,0,int(1E2*get_fa(countrycode)))]
	#
	try: 
		percentiles_urban  = perc_with_spline(income_proj[urb],weights_proj_tot[urb],np.arange(0,1,0.01))
		indicators['avg_income_urban'] = [np.average(income_proj[urb],weights=weights_proj_tot[urb])]
		indicators['avg_income_bott20_urban']    = [poverty_indic(percentiles_urban,0,20)]
	except: pass
	try: 
		percentiles_rural  = perc_with_spline(income_proj[rur],weights_proj_tot[rur],np.arange(0,1,0.01))
		indicators['avg_income_bott20_rural'] = [poverty_indic(percentiles_rural,0,20)]
		indicators['avg_income_rural']   = [np.average(income_proj[rur],weights=weights_proj_tot[rur])]
	except: pass
	#
	quintilescum               = wp(income_proj,weights_proj_tot,[0.2,0.4,1])
	quintilespc                = quintilescum/quintilescum[-1]
	indicators['GDP']          = [GDP(income_proj,weights_proj_tot)]

	#indicators['incbott10']    = [poverty_indic(percentiles,0,10)]
	#indicators['incbott40']    = [poverty_indic(percentiles,0,40)]
	#
	#
	indicators['incQ2']        = [poverty_indic(percentiles,20,40)]
	indicators['incQ3']        = [poverty_indic(percentiles,40,60)]
	indicators['incQ4']        = [poverty_indic(percentiles,60,80)]
	indicators['incQ5']        = [poverty_indic(percentiles,80,100)]
	indicators['quintilecum1'] = [quintilescum[0]]
	indicators['quintilecum2'] = [quintilescum[1]]
	indicators['quintilepc1']  = [quintilespc[0]]
	indicators['quintilepc2']  = [quintilespc[1]]
	#
	if ini_year is not None: indicators['start_year'] = ini_year
	#
	indicators['gini']         = [gini(income_proj,weights_proj_tot)]
	indicators['tot_pop']      = [sum(weights_proj*futurehhframe['nbpeople'])]
	#
	indicators['pop_190']      = [poor_people(income_proj,weights_proj_tot,1.90*data2day)]
	indicators['pop_320']      = [poor_people(income_proj,weights_proj_tot,3.20*data2day)]
	indicators['pop_550']      = [poor_people(income_proj,weights_proj_tot,5.50*data2day)]
	indicators['pop_1000']      = [poor_people(income_proj,weights_proj_tot,10.00*data2day)]
	#
	indicators['pop_190_urban'] = [poor_people(income_proj[urb],weights_proj_tot[urb],1.90*data2day)]
	indicators['pop_320_urban'] = [poor_people(income_proj[urb],weights_proj_tot[urb],3.20*data2day)]
	indicators['pop_550_urban'] = [poor_people(income_proj[urb],weights_proj_tot[urb],5.50*data2day)]
	indicators['pop_190_rural'] = [poor_people(income_proj[rur],weights_proj_tot[rur],1.90*data2day)]
	indicators['pop_320_rural'] = [poor_people(income_proj[rur],weights_proj_tot[rur],3.20*data2day)]
	indicators['pop_550_rural'] = [poor_people(income_proj[rur],weights_proj_tot[rur],5.50*data2day)]
	#
	indicators['pop_190_service'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat1workers+cat2workers)'),1.90*data2day)]
	indicators['pop_190_agricul'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat3workers+cat4workers)'),1.90*data2day)]
	indicators['pop_190_manufac'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat5workers+cat6workers)'),1.90*data2day)]
	#
	indicators['pop_320_service'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat1workers+cat2workers)'),3.20*data2day)]
	indicators['pop_320_agricul'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat3workers+cat4workers)'),3.20*data2day)]
	indicators['pop_320_manufac'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat5workers+cat6workers)'),3.20*data2day)]
	#
	indicators['pop_550_service'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat1workers+cat2workers)'),5.50*data2day)]
	indicators['pop_550_agricul'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat3workers+cat4workers)'),5.50*data2day)]
	indicators['pop_550_manufac'] = [poor_people(income_proj,futurehhframe.eval('weight*(cat5workers+cat6workers)'),5.50*data2day)]
	#
	indicators['gap_190']      = [poverty_gap(income_proj,weights_proj_tot,1.90*data2day)]
	indicators['gap_320']      = [poverty_gap(income_proj,weights_proj_tot,3.20*data2day)]
	indicators['gap_550']      = [poverty_gap(income_proj,weights_proj_tot,5.50*data2day)]
	#
	indicators['gap_190_service'] = [poverty_gap(income_proj[sv],weights_proj_tot[sv],1.90*data2day)]
	indicators['gap_190_agricul'] = [poverty_gap(income_proj[ag],weights_proj_tot[ag],1.90*data2day)]
	indicators['gap_190_manufac'] = [poverty_gap(income_proj[mn],weights_proj_tot[mn],1.90*data2day)]
	indicators['gap_320_service'] = [poverty_gap(income_proj[sv],weights_proj_tot[sv],3.20*data2day)]
	indicators['gap_320_agricul'] = [poverty_gap(income_proj[ag],weights_proj_tot[ag],3.20*data2day)]
	indicators['gap_320_manufac'] = [poverty_gap(income_proj[mn],weights_proj_tot[mn],3.20*data2day)]
	indicators['gap_550_service'] = [poverty_gap(income_proj[sv],weights_proj_tot[sv],5.50*data2day)]
	indicators['gap_550_agricul'] = [poverty_gap(income_proj[ag],weights_proj_tot[ag],5.50*data2day)]
	indicators['gap_550_manufac'] = [poverty_gap(income_proj[mn],weights_proj_tot[mn],5.50*data2day)]
	#
	#indicators['belowpovline'] = [poor_people(income_proj,weights_proj_tot,povline*data2day)]
	#indicators['below2']       = [poor_people(income_proj,weights_proj_tot,2*data2day)]
	#indicators['below4']       = [poor_people(income_proj,weights_proj_tot,4*data2day)]
	#indicators['below6']       = [poor_people(income_proj,weights_proj_tot,6*data2day)]
	#indicators['below8']       = [poor_people(income_proj,weights_proj_tot,8*data2day)]
	#indicators['below10']      = [poor_people(income_proj,weights_proj_tot,10*data2day)]
	#indicators['gap2']         = [poverty_gap(income_proj,weights_proj_tot,2*data2day)]
	#indicators['gap4']         = [poverty_gap(income_proj,weights_proj_tot,4*data2day)]
	#indicators['gap6']         = [poverty_gap(income_proj,weights_proj_tot,6*data2day)]
	#indicators['gap8']         = [poverty_gap(income_proj,weights_proj_tot,8*data2day)]
	#indicators['gap10']        = [poverty_gap(income_proj,weights_proj_tot,10*data2day)]
	#
	all_employed = 'weight*(cat1workers+cat2workers+cat3workers+cat4workers+cat5workers+cat6workers)'
	all_employed_df = futurehhframe[['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers']].sum(axis=1)>0

	indicators['decent_jobs_urban'] = futurehhframe[(urb)&(income_proj>1.90*data2day)].eval(all_employed).sum()
	indicators['decent_jobs_rural'] = futurehhframe[(rur)&(income_proj>1.90*data2day)].eval(all_employed).sum()
	indicators['decent_jobs_service'] = futurehhframe.loc[income_proj>1.90*data2day].eval('weight*(cat1workers+cat2workers)').sum()
	indicators['decent_jobs_agricul'] = futurehhframe.loc[income_proj>1.90*data2day].eval('weight*(cat3workers+cat4workers)').sum()
	indicators['decent_jobs_manufac'] = futurehhframe.loc[income_proj>1.90*data2day].eval('weight*(cat5workers+cat6workers)').sum()
	#
	indicators['working_poor_urban'] = futurehhframe[(urb)&(income_proj<=1.90*data2day)].eval(all_employed).sum()
	indicators['working_poor_rural'] = futurehhframe[(rur)&(income_proj<=1.90*data2day)].eval(all_employed).sum()
	indicators['working_poor_service'] = futurehhframe.loc[income_proj<=1.90*data2day].eval('weight*(cat1workers+cat2workers)').sum()
	indicators['working_poor_agricul'] = futurehhframe.loc[income_proj<=1.90*data2day].eval('weight*(cat3workers+cat4workers)').sum()
	indicators['working_poor_manufac'] = futurehhframe.loc[income_proj<=1.90*data2day].eval('weight*(cat5workers+cat6workers)').sum()
	#
	indicators['ppl_in_hh_working_poor_urban'] = futurehhframe.loc[(urb)&(income_proj<=1.90*data2day)&(working),'totweight'].sum()
	indicators['ppl_in_hh_working_poor_rural'] = futurehhframe.loc[(rur)&(income_proj<=1.90*data2day)&(working),'totweight'].sum()
	#
	indicators['pop_sv_unskilled'] = futurehhframe[['cat1workers','weight']].prod(axis=1).sum()
	indicators['pop_sv_skilled'] = futurehhframe[['cat2workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_unskilled'] = futurehhframe[['cat3workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_skilled'] = futurehhframe[['cat4workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_unskilled'] = futurehhframe[['cat5workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_skilled'] = futurehhframe[['cat6workers','weight']].prod(axis=1).sum()
	#
	indicators['pop_sv_unskilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat1workers','weight']].prod(axis=1).sum()
	indicators['pop_sv_skilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat2workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_unskilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat3workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_skilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat4workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_unskilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat5workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_skilled_urban'] = futurehhframe.loc[futurehhframe.urban!=0,['cat6workers','weight']].prod(axis=1).sum()
	#
	indicators['pop_sv_unskilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat1workers','weight']].prod(axis=1).sum()
	indicators['pop_sv_skilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat2workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_unskilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat3workers','weight']].prod(axis=1).sum()
	indicators['pop_ag_skilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat4workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_unskilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat5workers','weight']].prod(axis=1).sum()
	indicators['pop_mn_skilled_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat6workers','weight']].prod(axis=1).sum()
	#
	indicators['unemployed'] = futurehhframe[['cat7workers','weight']].prod(axis=1).sum()
	indicators['unemployed_rural'] = futurehhframe.loc[futurehhframe.urban==0,['cat7workers','weight']].prod(axis=1).sum()
	# 
	indicators['sv_income_pc'] = (income_proj[sv]*futurehhframe.ix[sv,'totweight']).sum()/futurehhframe.ix[sv,'totweight'].sum()
	indicators['sv_unskilled_income_pc'] = (income_proj[sv_u]*futurehhframe.ix[sv_u,'totweight']).sum()/futurehhframe.ix[sv_u,'totweight'].sum()
	indicators['sv_skilled_income_pc'] = (income_proj[sv_s]*futurehhframe.ix[sv_s,'totweight']).sum()/futurehhframe.ix[sv_s,'totweight'].sum()
	#
	indicators['ag_income_pc'] = (income_proj[ag]*futurehhframe.ix[ag,'totweight']).sum()/futurehhframe.ix[ag,'totweight'].sum()
	indicators['ag_unskilled_income_pc'] = (income_proj[ag_u]*futurehhframe.ix[ag_u,'totweight']).sum()/futurehhframe.ix[ag_u,'totweight'].sum()
	indicators['ag_skilled_income_pc'] = (income_proj[ag_s]*futurehhframe.ix[ag_s,'totweight']).sum()/futurehhframe.ix[ag_s,'totweight'].sum()
	#
	indicators['mn_income_pc'] = (income_proj[mn]*futurehhframe.ix[mn,'totweight']).sum()/futurehhframe.ix[mn,'totweight'].sum()
	indicators['mn_unskilled_income_pc'] = (income_proj[mn_u]*futurehhframe.ix[mn_u,'totweight']).sum()/futurehhframe.ix[mn_u,'totweight'].sum()
	indicators['mn_skilled_income_pc'] = (income_proj[mn_s]*futurehhframe.ix[mn_s,'totweight']).sum()/futurehhframe.ix[mn_s,'totweight'].sum()
	#
	indicators['nonag_income_pc'] = (income_proj[~ag]*futurehhframe.ix[~ag,'totweight']).sum()/futurehhframe.ix[~ag,'totweight'].sum()
	#
	#percentiles_ag = perc_with_spline(income_proj[ag],weights_proj_tot[ag],np.arange(0,1,0.01))
	#percentiles_nonag = perc_with_spline(income_proj[~ag],weights_proj_tot[~ag],np.arange(0,1,0.01))
	#
	#
	#for pp in np.arange(0,100):
	#	indicators['percentiles_ag_{}'.format(pp)] = percentiles_ag[pp]
	#	indicators['percentiles_nonag_{}'.format(pp)] = percentiles_nonag[pp]

	#urb_poor = (futurehhframe['urban']>0)&(futurehhframe['Y']<=365*1.9)
	#rur_poor = (futurehhframe['urban']==0)&(futurehhframe['Y']<=365*1.9)
	indicators['gap_190_urban'] = [poverty_gap(income_proj[urb],weights_proj_tot[urb],1.90*data2day)]
	indicators['gap_320_urban'] = [poverty_gap(income_proj[urb],weights_proj_tot[urb],3.20*data2day)]
	indicators['gap_550_urban'] = [poverty_gap(income_proj[urb],weights_proj_tot[urb],5.50*data2day)]
	indicators['gap_190_rural'] = [poverty_gap(income_proj[rur],weights_proj_tot[rur],1.90*data2day)]
	indicators['gap_320_rural'] = [poverty_gap(income_proj[rur],weights_proj_tot[rur],3.20*data2day)]
	indicators['gap_550_rural'] = [poverty_gap(income_proj[rur],weights_proj_tot[rur],5.50*data2day)]
	#
	indicators['childrenag']  = sum(futurehhframe.ix[ag,'children']*futurehhframe.ix[ag,'weight'])
	indicators['childrenonag']= sum(futurehhframe.ix[~ag,'children']*futurehhframe.ix[~ag,'weight'])
	indicators['peopleag']    = sum(futurehhframe.ix[ag,'nbpeople']*futurehhframe.ix[ag,'weight'])
	indicators['peoplenonag'] = sum(futurehhframe.ix[~ag,'nbpeople']*futurehhframe.ix[~ag,'weight'])
	#
	df = read_csv('exposed_pop/{}.csv'.format(countrycode)).set_index('hhid')
	futurehhframe.index.name = 'hhid'
	
	_hhdf = merge(futurehhframe.reset_index(),df.reset_index(),on='hhid')
	indicators['exposed_pop_Q1'] = sum(_hhdf.loc[_hhdf['Y']<=float(indicators['avg_income_bott20'])*365,'totweight'])
	indicators['exposed_pop_ifrc_fa'] = sum(_hhdf.loc[_hhdf['Y']<=float(indicators['avg_income_ifrc_fa'])*365,'totweight'])
	indicators['exposed_pop_190'] = sum(_hhdf.loc[_hhdf['190']==True,'totweight'])
	indicators['exposed_pop_320'] = sum(_hhdf.loc[_hhdf['320']==True,'totweight'])
	indicators['exposed_pop_1000'] = sum(_hhdf.loc[_hhdf['1000']==True,'totweight'])


	# quintilesag               = wp(reshape_data(income_proj[ag]),reshape_data(weights_proj_tot[ag]),[0.2,1],cum=True)
	# quintilesnonag            = wp(reshape_data(income_proj[~ag]),reshape_data(weights_proj_tot[~ag]),[0.2,1],cum=True)
	# quintilesagpc             = quintilesag/quintilesag[-1]
	# quintilesnonagpc          = quintilesnonag/quintilesnonag[-1]
	
	# indicators['incsharebott20ag'] = quintilesagpc[0]
	# indicators['incsharebott20nonag'] = quintilesnonagpc[0]
		
	# indicators['poorag']      = [poor_people(income_proj[ag],weights_proj_tot[ag],povline*data2day)]
	# indicators['poornonag']   = [poor_people(income_proj[~ag],weights_proj_tot[~ag],povline*data2day)]
	
	# indicators['incbott20ag']   = [poverty_indic_spec(income_proj[ag],weights_proj_tot[ag],percentiles[20])]
	# indicators['incbott20nonag']   = [poverty_indic_spec(income_proj[~ag],weights_proj_tot[~ag],percentiles[20])]
	
	indicators['avg_income_ag'] = [np.average(income_proj[ag],weights=weights_proj_tot[ag])]
	indicators['avg_income_nonag']   = [np.average(income_proj[~ag],weights=weights_proj_tot[~ag])]

	_fa = [('fa20_earthquake','EARTHQUAKE'),
	       ('fa20_wind','WIND'),
	       ('fa20_surge','STORM_SURGE'),
	       ('fa20_tsunami','TSUNAMI'),
	       ('fa20_riverflood','Riverine_floods')]
		
	_rps = read_excel('./data/disasters/gar_for_model_rp=20.xlsx',index_col='ISO',sheetname='rp20_fa').ix[countrycode]

	for _out,_in in _fa:
		indicators[_out] = 1E2*_rps.ix[_in]/2E1


	return indicators
	
	
