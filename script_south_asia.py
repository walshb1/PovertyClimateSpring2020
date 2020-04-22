import pandas as pd
import matplotlib.pyplot as plt
import os
from lib_for_growth_model import *
from decimal import Decimal
import numpy as np
from scipy.stats import gaussian_kde
from pandas_helper import *

from lib_for_plot import *
import seaborn as sns

sns.set_context("poster",rc={"font.size": 18})
sns.set_style("whitegrid")
import matplotlib.gridspec as gridspec

codes = read_csv('wbccodes2014.csv')
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
	
from lib_for_analysis import *

def anova_table(varin,data,experiments_cols):
	formula = varin+" ~ " + "+".join(experiments_cols)
	olsmodel=ols(formula,data=data).fit()
	table=anova_lm(olsmodel)
	table['sum_sq_pc']=table['sum_sq']/table['sum_sq'].sum()
	table=table.sort(['sum_sq'],ascending=False)
	return table['sum_sq_pc']
	
def drivers_matrix(var,hop,myinputs):
	impacts=['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']
	df = pd.DataFrame(index=impacts,columns=myinputs+["Residual"])
	for switch in impacts:
		to_plot = hop[hop[switch]].reset_index()
		to_plot["ccint"] = to_plot["ccint"].replace(dict({0:'Low',1:'High'}))
		t = anova_table(var,to_plot,myinputs)
		df.loc[switch,:]=t
	return df


codes['country'] = codes.country.apply(correct_countrycode)

nameofthisround = 'sept2016_separate_impacts'
model = os.getcwd()
bau_folder   = "{}/baselines_{}/".format(model,nameofthisround)
cc_folder    = "{}/with_cc_{}/".format(model,nameofthisround)


list_bau = os.listdir(bau_folder)
list_cc  = os.listdir(cc_folder)
bau = pd.DataFrame()
cc  = pd.DataFrame()

for file in list_bau:
    bau = bau.append(pd.read_csv(bau_folder+file).drop('Unnamed: 0',axis=1))
    
for file in list_cc:
    temp = pd.read_csv(cc_folder+file).drop('Unnamed: 0',axis=1)
    if "disasters" not in file:
        print(file)
        temp = temp.ix[temp.switch_disasters==0,:]
    cc = cc.append(temp)

bau = bau.drop_duplicates(['country', 'scenar', 'ssp'])
cc = cc.dropna()
bau = bau.dropna()

for col_switch in ['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']:
    cc.ix[cc[col_switch],"switch"] = col_switch

cc['issp5'] = cc.ssp=='ssp5'
bau['issp5'] = bau.ssp=='ssp5'
bau['countryname'] = bau.country.replace(codes.set_index('country').country_name)
cc['countryname'] = cc.country.replace(codes.set_index('country').country_name)


graphs_folder = "country_results/"
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)
	
results_by_country = "results_by_country/"
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)

for cc in bau.country.unique():
	bau[bau.country==selectedcountry].to_csv("{}bau_{}".format(results_by_country,cc),ignore_index=True)
	cc[cc.country==selectedcountry].to_csv("{}cc_{}".format(results_by_country,cc),ignore_index=True)

	
for selectedcountry in list(codes[codes.wbregion=='SAS'].country):
	
	if sum(bau.country==selectedcountry)==0:
		continue

	bau_c = bau[bau.country==selectedcountry]
	cc_c  = cc[cc.country==selectedcountry]

	hop = cc_c.set_index(['scenar', 'ssp','ccint','switch'])
	hip = broadcast_simple(bau_c.set_index(['scenar', 'ssp']),hop.index)

	hop['below125diff'] = (hop['below125']-hip['below125'])
	hop['below125diff_pc_of_pop'] = (hop['below125']-hip['below125'])/hop['tot_pop']
	hop['below125diff_relative_change'] = (hop['below125']-hip['below125'])/hip['below125']


	hop['incbott40diff'] = (hop.incbott40-hip.incbott40)/hip.incbott40
	hop['avincomediff'] = (hop.avincome-hip.avincome)/hip.avincome


	font = {'family' : 'sans serif',
		'size'   : 16}
	plt.rc('font', **font)

	titles = ["Agriculture\n revenues","Labor\nproductivity","Food prices","Disasters","Health"]

	fig = plt.figure(figsize=(12,5))
	gs1 = gridspec.GridSpec(1, 5)
	gs1.update(wspace=0.025, hspace=0.05)
	i=0

	for switch in ['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']:
		if abs(10**(-6)*hop.below125diff.min())>1:
			rescale = 10**(-6)
			metrics = "million"
		elif abs(10**(-3)*hop.below125diff.min())>1:
			rescale = 10**(-3)
			metrics = "thousand"
		else:
			rescale = 1
			metrics =""

		ax = fig.add_subplot(gs1[i])
		ax.get_xaxis().set_ticks([])
		ax.set_ylim([rescale*hop.below125diff.min(),rescale*hop.below125diff.max()])
		spine_and_ticks(ax)
		#ax.set_aspect('equal')

		to_plot = pd.DataFrame(hop.below125diff.unstack("switch")[switch].reset_index())
		to_plot["ccint"] = to_plot["ccint"].replace(dict({0:'Low',1:'High'}))
		to_plot[switch] = rescale*to_plot[switch]

		to_plot['rien']=""

		sns.boxplot(x='rien',y=switch,data=to_plot,ax=ax,palette="Set2")
		plt.xlabel(titles[i])
		sns.despine(bottom=True)
		if i==0:
			plt.ylabel("People living in extreme poverty\ndue to climate change ({}, 2030)".format(metrics))
		else:
			plt.ylabel("")
			ax.set_yticklabels([])
		i+=1
		
	plt.savefig("{}all_impacts_extrpoors_{}".format(graphs_folder,selectedcountry),bbox_inches="tight",dpi=200)


	font = {'family' : 'sans serif',
		'size'   : 16}
	plt.rc('font', **font)

	titles = ["Agriculture\n revenues","Labor\nproductivity","Food prices","Disasters","Health"]

	fig = plt.figure(figsize=(12,5))
	gs1 = gridspec.GridSpec(1, 5)
	gs1.update(wspace=0.025, hspace=0.05)
	i=0

	for switch in ['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']:
		ax = fig.add_subplot(gs1[i])
		ax.get_xaxis().set_ticks([])
		ax.set_ylim([np.min(-100*hop.incbott40diff),np.max(-100*hop.incbott40diff)])
		spine_and_ticks(ax)
		#ax.set_aspect('equal')

		to_plot = pd.DataFrame(hop.incbott40diff.unstack("switch")[switch].reset_index())
		to_plot["ccint"] = to_plot["ccint"].replace(dict({0:'Low',1:'High'}))
		to_plot[switch] = -100*to_plot[switch]

		to_plot['prout']=""

		sns.boxplot(x='prout',y=switch,data=to_plot,ax=ax,palette="Set2")
		plt.xlabel(titles[i])
		sns.despine(bottom=True)
		if i==0:
			plt.ylabel("Income loss of the bottom 40%\ndue to climate change (%)")
		else:
			plt.ylabel("")
			ax.set_yticklabels([])
		i+=1
		
	plt.savefig("{}all_impacts_bott40_{}".format(graphs_folder,selectedcountry),bbox_inches="tight",dpi=200)

	myinputs = ['shareag','sharemanu', 'shareemp', 'grserv', 'grag', 'grmanu', 'skillpserv','skillpag', 'skillpmanu', 'p', 'b','ccint','issp5','voice']

	drivers_country_poor_pc = drivers_matrix('below125diff_pc_of_pop',hop,myinputs)
	drivers_country_incbott40 = drivers_matrix('incbott40diff',hop,myinputs)

	aa = dict({'switch_ag_rev':"Agriculture\n revenues","switch_temp":"Labor\nproductivity","switch_ag_prices":"Food prices","switch_disasters":"Disasters","switch_health":"Health"})

	bb = dict({'shareag':'Share of workers in ag','sharemanu':'Share of workers in manu','shareemp':'Employment rate','grserv':'Growth in services','grag':'Growth in ag','grmanu':'Growth in manu','skillpserv':'Skill premium services','skillpag':'Skill premium ag','skillpmanu':'Skill premium manu','p':'Pensions','b':'Redistribution','ccint':'Climate change impacts','issp5':'Demographics','voice':'Governance','Residual':'Nonlinear interactions'})

	to_plot = drivers_country_incbott40.rename(columns=bb).rename(index=aa)
	cci = to_plot['Climate change impacts']
	to_plot.drop(labels=['Climate change impacts'], axis=1,inplace = True)
	to_plot.insert(0, 'Climate change impacts', cci)
	ax=to_plot.drop([col for col, val in to_plot.sum(axis=0).iteritems() if val < 0.02], axis=1).plot(kind='bar',stacked=True,rot=0,colormap="Accent")
	ax.legend(bbox_to_anchor=(1.35, 1.))
	plt.title("Sources of uncertainy for\nincome loss of the bottom 40% because of climate change in 2030")
	plt.ylabel("Share of uncertainty explained")
	plt.savefig("{}uncertainty_bott40_{}".format(graphs_folder,selectedcountry),bbox_inches="tight",dpi=200)


	to_plot = drivers_country_poor_pc.rename(columns=bb).rename(index=aa)
	cci = to_plot['Climate change impacts']
	to_plot.drop(labels=['Climate change impacts'], axis=1,inplace = True)
	to_plot.insert(0, 'Climate change impacts', cci)
	ax=to_plot.drop([col for col, val in to_plot.sum(axis=0).iteritems() if val < 0.02], axis=1).plot(kind='bar',stacked=True,rot=0,colormap="Accent")
	ax.legend(bbox_to_anchor=(1.35, 1.))
	plt.title("Sources of uncertainy for\nadditional people in extreme poverty because of climate change in 2030")
	plt.ylabel("Share of uncertainty explained")
	plt.savefig("{}uncertainty_extrpoor_{}".format(graphs_folder,selectedcountry),bbox_inches="tight",dpi=200)




