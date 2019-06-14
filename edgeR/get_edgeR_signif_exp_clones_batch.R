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

diff_group_yf<-c("d0","d0","d15","d15","d45","d45","d7","d7","dp0","dp0")

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
YF_all<-import_folder_dt_fread("/Volumes/Data/YF_assemble_1/mixcr/Fulls/")
#susbset for F replicates and pass it through edger pipeline
GS_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("GS",names(YF_all))])
Kar_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Kar",names(YF_all))])
Luci_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Luci",names(YF_all))])
Azh_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Azh",names(YF_all))])
Yzh_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("Yzh",names(YF_all))])
KB_exp<-standart_experiment_DE_YF(YF_all[grepl("F",names(YF_all))&grepl("KB",names(YF_all))])
exp_list<-list(Azh=Azh_exp,Yzh=Yzh_exp,KB=KB_exp,GS=GS_exp,Kar=Kar_exp,Luci=Luci_exp)

#select significantly expanded clones and filter them by log2FC threshold 
top15_3<-lapply(exp_list,function(x){topTags(exactTest(x,pair = c("d0","d15"),dispersion = "trended"),n=5000,p.value = 0.01)$table})
top15_3<-lapply(top15_3,function(x){x$CDR3nt<-row.names(x);x})
top15_3_sign<-lapply(top15_3,function(x)x[x$logFC>5,])