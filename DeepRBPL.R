#This text file contain the steps to follow for reproducing the 
#developed computational approach DeepRBPL####

#There are mainly 3 steps involved to implement the proposed approach#

##############################################FIRST STEP#########################

#Genertae the .pssm profile by running PSI-blast on NR50 database(Non-redundant at 50% sequence identity) using R-3.6.0 or any other vesrion#

#1. Download the BLAST software in your system

#2. Set the working directory for the BLAST as follows 

setwd("C:/Program Files/NCBI/blast-2.6.0+/bin") # set the bin folder of the BLAST as working directory

#3. generate the .pssm file using the NR50 dataset using the following command in R

shell("makeblastdb -in C:/Users/USER/Desktop/PSSM_feature/NR50.fasta -dbtype prot -out C:/Users/USER/Desktop/PSSM_feature/train")

# Here, "C:/Users/USER/Desktop/PSSM_feature" is the path of the NR50.fasta and 
# "C:/Users/USER/Desktop/PSSM_feature" is the path of the file where .pssm file would be saved in the name of train
#There will be three .pssm files such as train.phr, train.psq and train.pin

###########################################SECOND STEP################################
# The second step involves the generation of features of the sequence dataset (training and testing) using R-4.0.2

#1. install R-4.0.2
# Install the packages "Biostrings" (Biocondutor) and "PSSMCOOL" (CRAN)
library(Biostrings)
library(PSSMCOOL)
# put all the sequence dataset in the folder where the .pssm profiles have been put
#Set the working directory 

setwd("C:/Users/USER/Desktop/PSSM_feature")
# run one sequence at a time to generate the feature. So to generate the feature for more sequence the code should be in a loop

# read the protein sequence data set containing more than one sequence in fasta format.  

dat <- readAAStringSet("sequence_data.fasta")

#generate the AADP, EEDP, MEDP and KSBG (k-separated bigram)

pssm_AADP <- matrix(0, nrow=length(x), ncol=420)#AADP PSSM
pssm_EEDP <- matrix(0, nrow=length(x), ncol=400)#EEDP_PSSM
pssm_MEDP <- matrix(0, nrow=length(x), ncol=420)#MEDP_PSSM
pssm_KSBG <- matrix(0, nrow=length(x), ncol=400)#kseparated bigram_PSSM


for(i in 1:length(dat)){

writeXStringSet(dat[i],"query.fasta")
setwd("C:/Program Files/NCBI/blast-2.6.0+/bin") # set the bin folder of the BLAST as working directory
shell("psiblast -query C:/Users/USER/Desktop/PSSM_feature/query.fasta -db C:/Users/USER/Desktop/PSSM_feature/train -num_iterations 3 -evalue 0.001 -out_ascii_pssm C:/Users/USER/Desktop/PSSM_feature/protein.pssm")

setwd("C:/Users/USER/Desktop/PSSM_feature")

pssm_AADP[i,]<-as.matrix(suppressWarnings(as.numeric(aadp_pssm("protein.pssm"))), ncol=420, nrow=1)

pssm_EEDP[i,]<-as.matrix(suppressWarnings(as.numeric(EDP_EEDP_MEDP("protein.pssm")[[2]])), ncol=400, nrow=1)

pssm_MEDP[i,]<-as.matrix(suppressWarnings(as.numeric(EDP_EEDP_MEDP("protein.pssm")[[3]])), ncol=420, nrow=1)

pssm_KSBG[i,]<-as.matrix(suppressWarnings(as.numeric(k_separated_bigrams_pssm("protein.pssm", k=1))), ncol=400, nrow=1)

}

feature <- cbind(pssm_AADP, pssm_EEDP, pssm_MEDP, pssm_KSBG)
write.table(featureP,"feature.txt", row.names=FALSE, col.names=FALSE, sep="\t")

#if using the RBP sequence dataset, save the file as "positive_feature.txt"
#if using the non-RBP sequence dataset, save the file as "negative_feature.txt"

#The above encoded four feature sets would be saved in  the path "C:/Users/USER/Desktop/PSSM_feature".


#####################################THIRD STEP###################################





