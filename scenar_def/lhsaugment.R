lhsssample=read.csv("lhs-table-300-11uncertainties.csv",head=TRUE,sep=",")
library(lhs)

lhsmat=data.matrix(lhsssample)

newlhs=optAugmentLHS(lhsmat, m=300, mult=2)

write.csv(file="lhs-table-600-11uncertainties.csv",x=newlhs)