import sys
from pandas import read_excel,DataFrame,pivot_table
import os

redistrib     = True
day2data      = 30
year          = 2030
ini_year      = 2007

nameofthisround = 'withskillp_jan23'

food = read_excel("CC_MTG_impacts_15apr15_wDM_v2.xlsx","Data_CC_MTG_impacts_15apr15")

select = (food.Var=="XPRP")&((food.RCP=="No CC")|(food.RCP=="RCP8.5"))&(food.Year==2030)&(food.Policy=='NoMitig')&(food.GCM!="Avg")&(food.Item=="CRP")

def f(x):
	return (x.set_index("GCM").Val/x.ix[x.GCM=="None","Val"].values[0]-1)
	# return x.ix[x.GCM=="None","Val"].values[0]

price1=food.ix[select,:].groupby(["Reg","Macro"]).apply(f)
price1=pivot_table(price1.reset_index().drop(["None"],axis=1),columns=["Reg","Macro"]).reset_index()

out = DataFrame(columns=['min','max','ssp'])
for ssp in ["SSP4","SSP5"]:
	out1 = DataFrame(columns=['min','max','ssp'])
	out1.loc["SSA",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['Africa'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['Africa'],ssp]
	out1.loc["MNA",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['MidEastNorthAfr'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['MidEastNorthAfr'],ssp]
	out1.loc["EAP",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['EastAsiaPac'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['EastAsiaPac'],ssp]
	out1.loc["LAC",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['LatinAmericaCarib'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['LatinAmericaCarib'],ssp]
	out1.loc["SAS",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['SouthAsia'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['SouthAsia'],ssp]
	out1.loc["ECA",:]=[price1.ix[price1.Macro=="SSP4"].groupby("Reg").apply(lambda x:x[0].min())['EurCentAsia'],price1.ix[price1.Macro==ssp].groupby("Reg").apply(lambda x:x[0].max())['EurCentAsia'],ssp]
	out = out.append(out1)

out.to_csv("price_increase_2030.csv")


