library(lhs)

numCases = 100
numUncertainties=11

#experiment=optimumLHS(n=numCases,k=numUncertainties)
experiment=maximinLHS(n=numCases,k=numUncertainties,dup=1)


experiment=data.frame(experiment)

write.csv(file="lhsmaximin-table-100-11uncertainties.csv",x=experiment)