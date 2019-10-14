# for max edger

#dependencies
library(data.table)
library(edgeR)
#neccesary functions
import_folder_dt_fread<-function(path){
  fnames<-list.files(path = path,full.names = F)
  dflist<-lapply(list.files(path = path,full.names = T),fread)
  names(dflist)<-gsub("_.txt","",fnames)
  lapply(dflist,function(x){setnames(x,names(x)[1:4],c("Rank","Read.count","Read.proportion","CDR3.nucleotide.sequence"))})
  lapply(dflist,function(x){setnames(x,"AA. Seq. CDR3","CDR3.amino.acid.sequence")})
  dflist<-lapply(dflist,function(x){x[,.(Read.count,Read.proportion,CDR3.nucleotide.sequence,CDR3.amino.acid.sequence,Rank),]})
  lapply(dflist,setkey,CDR3.nucleotide.sequence)
  dflist
}

merge_dt_list<-function(DTlist,colname="Read.proportion",bycol="CDR3.nucleotide.sequence"){
  DTlist<-lapply(DTlist,function(x)x[,c(bycol,colname),with=F])
  #change names
  lapply(names(DTlist),function(x)setnames(DTlist[[x]],colname,paste0(x,".",colname)))
  DTlistm<-rbindlist(DTlist,fill=T)
  print("Binded")
  DTlistm<-DTlistm[,lapply(.SD, sum,na.rm=T),bycol]
  print("Merged")
  DTlistm[, Sum := Reduce(`+`, .SD), .SDcols=grep(colname, names(DTlistm))][]
  print("Sums done")
  DTlistm[, Mean :=Sum/(length(grep(colname, names(DTlistm)))),][]
}

diff_group_yf<-c("d0","d0","d15","d15")#,"d45","d45","d7","d7","dp0","dp0")

standart_experiment_DE_YF<-function(DTlist,thres=4,grp=diff_group_yf){
  mdt<-merge_dt_list(DTlist = DTlist,bycol = "CDR3.nucleotide.sequence",colname = "Read.count")
  mdt<-mdt[Mean>thres,,]
  CDRs<-mdt[,CDR3.nucleotide.sequence,]
  mdt<-as.matrix(as.data.frame(mdt)[,-c(1,ncol(mdt),ncol(mdt)-1)])
  row.names(mdt)<-CDRs
  colnames(mdt)<-names(DTlist)
  y<-DGEList(counts = mdt,group=grp)
  y<-calcNormFactors(y)
  y<-estimateDisp(y)
  y
}

#load data
YF_all<-import_folder_dt_fread("../../data/Yellow_fever/prepostvaccine/test/")
#susbset for F replicates and pass it through edger pipeline
P1_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("P1",names(YF_all))])
Q1_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Q1",names(YF_all))])
Q2_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Q2",names(YF_all))])
S2_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("S2",names(YF_all))])
S1_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("S1",names(YF_all))])
P2_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("P2",names(YF_all))])
exp_list<-list(S2=S2_exp,S1=S1_exp,P2=P2_exp,P1=P1_exp,Q1=Q1_exp,Q2=Q2_exp)

#select significantly expanded clones and filter them by log2FC threshold 
top15_3<-lapply(exp_list,function(x){topTags(exactTest(x,pair = c("d0","d15"),dispersion = "trended"),n=5000,p.value = 0.01)$table})
top15_3<-lapply(top15_3,function(x){x$CDR3nt<-row.names(x);x})
top15_3_sign<-lapply(top15_3,function(x)x[x$logFC>5,])

for (i in seq_along(top15_3_sign)){write.csv(top15_3_sign[names(top15_3_sign)[i]],file=names(top15_3_sign)[i])}