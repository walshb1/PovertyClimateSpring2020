import sys
from pandas import read_excel,DataFrame,pivot_table,concat
import os
import numpy as np

redistrib     = True
day2data      = 30
year          = 2030
ini_year      = 2007

nameofthisround = 'withskillp_jan23'

food = read_excel("CC_MTG_impacts_15apr15_wDM_v2.xlsx","Data_CC_MTG_impacts_15apr15")

selectall = ((food.RCP=="No CC")|(food.RCP=="RCP8.5")|(food.RCP=="RCP8.5*"))&(food.Year==2030)&(food.Policy=='NoMitig')&(food.GCM!="Avg")&(food.Item=="CRP")
selectq = (food.Var=="PROD")&(food.Unit=="1000 t")
selectp = (food.Var=="XPRP")

inter = food.ix[selectall&(selectq|selectp),:]
inter = inter.drop(["Year","Item","Policy","ID","Unit"],axis=1).set_index(['Reg','Macro','GCM','Var','RCP']).unstack('Var')['Val'].reset_index()
inter["PQ"] = inter.XPRP*inter.PROD

hop=inter.set_index(["Reg","Macro"])
hop[["prodd","priced","pqd"]]=hop[["PROD","XPRP","PQ"]]/hop.ix[hop.GCM=="None",["PROD","XPRP","PQ"]]-1
hop=hop.replace("None",np.nan).dropna().reset_index()

hop = hop.replace("Africa","SSA")
hop = hop.replace("MidEastNorthAfr","MNA")
hop = hop.replace("EastAsiaPac","EAP")
hop = hop.replace("LatinAmericaCarib","LAC")
hop = hop.replace("SouthAsia","SAS")
hop = hop.replace("EurCentAsia","ECA")
hop = hop.replace("WestEurope",np.nan)
hop = hop.replace("World",np.nan)
hop = hop.replace("NorthAmerica",np.nan)
hop = hop.replace("PacificDvd",np.nan).dropna()

minp = hop.groupby(["Reg","Macro"]).apply(lambda x:x.ix[x.priced==x.priced.min(),["prodd","priced","pqd"]]).reset_index().drop(["level_2"],axis=1)
maxp = hop.groupby(["Reg","Macro"]).apply(lambda x:x.ix[x.priced==x.priced.max(),["prodd","priced","pqd"]]).reset_index().drop(["level_2"],axis=1)

minp["min_max"] = "min"
maxp["min_max"] = "max"

out = concat([minp,maxp],axis=0)

out.to_csv("boundaries_food2.csv",index=False)



