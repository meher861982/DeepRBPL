# DeepRBPL
Deep leaning model for computational identification of plant-specific RNA binding proteins


Step-by-step procedure to reproduce the result of DeepRBPL using the users own dataset


1.	Save PSSM features of the positive dataset (RBP sequence) extracted by R code given in DeepRBPL.R as positive_feature.txt
2.	Save PSSM features of the negative dataset (non-RBP sequence) extracted by R code given in DeepRBPL.R as negative_feature.txt
3.	Create a new folder “Training_DeepRBPL” and keep all the three files, i.e., “positive_feature.txt”, “negative_feature.txt” and “Train_DeepRBPL.py” in that folder.
4.	Open a new terminal in the folder “Training_DeepRBPL”
5.	Run the code for training from command prompt: python Train_DeepRBPL.py positive_feature.txt negative_feature.txt
6.	Obtain result files from the generated output subfolder “DeepRBPL_Training_Results” within the parent folder “Training_DeepRBPL”.

