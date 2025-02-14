### Required Python modules and functions
from __future__ import division, print_function
import _ctypes
import csv
import numpy as np
import pandas as pd
import os, time, math, statistics, sys
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import ZeroPadding1D, TimeDistributed, Conv1D, MaxPooling1D, Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.utils import shuffle, resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, auc, precision_recall_curve, roc_auc_score, roc_curve
from scipy import stats
import h5py

def convert_tab_to_csv(input_file_path, output_file_path):
    with open(input_file_path, 'r', newline='') as input_file, open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',')
    
        for line in input_file:
            values = line.strip().split('\t')
            csv_writer.writerow(values)
    
    print(f"Conversion complete. CSV file saved at: {output_file_path}")

def main():
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python Train_DeepRBPL.py positive_feature.txt negative_feature.txt")
        return
    
    # Get input and output file names from command line arguments
    input_file_name = sys.argv[1]   
    output_file_name = 'positive_feature.csv'
    convert_tab_to_csv(input_file_name, output_file_name)
    
    input_file_name_1 = sys.argv[2]   
    output_file_name_1 = 'negative_feature.csv'
    convert_tab_to_csv(input_file_name_1, output_file_name_1)
    
    # Read the input CSV data
    Result_dir = 'DeepRBPL_Training_Results'
    Check_dir_Result = os.path.isdir(Result_dir)
    if not Check_dir_Result:
        os.makedirs(Result_dir)
        print("created folder : ", Result_dir)
    
    else:
        print(Result_dir, "folder already exists.")
    
    
    # Data input and Classification by DeepRBPL
    positive_data = pd.read_csv(output_file_name, header = None)
    positive_data1 = positive_data.iloc[:,]
    
    negative_data = pd.read_csv(output_file_name_1, header = None)
    negative_data1 = negative_data.iloc[:,]
    
    # Concatenating row-wise
    X = pd.concat([positive_data1, negative_data1], axis=0, ignore_index=True)
    
    Y1 = np.ones(positive_data1[0].count())
    Y2 = np.zeros(negative_data1[0].count())
    Y_numpy = np.concatenate([Y1, Y2], axis=0)
    Y = pd.DataFrame(Y_numpy)
    
    print (X)
    print (Y)

    Y1=Y.to_numpy()
    classes=np.unique(Y1)
    counter=0
    for i in classes:
        Y1[np.where(Y1==i)]=counter
        counter+=1
    features = X.shape[1]  
    
    print ("Total number of classes:",counter)
    print ("Total number of features:",features)
    
    ### CNN architecture
    def create_model(nb_classes,input_length):
        model = Sequential()
        model.add(Conv1D(5, 5, input_shape=(input_length, 1), padding="valid")) #input_dim
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Conv1D(10, 5, padding="valid"))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Flatten())
        ##
        ##MLP
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        #model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model
    
    
    
    ### CNN Model training and validation specifications with validation split within model fit

    def train_and_evaluate_model (model, train_feature, train_label, validation_feature, validation_label, nb_classes):
        global epochs
    
        train_feature = train_feature.reshape(train_feature.shape + (1,))
        train_label = np_utils.to_categorical(train_label, nb_classes)
        validation_label_bin = np_utils.to_categorical(validation_label, nb_classes)
        validation_feature = validation_feature.reshape(validation_feature.shape + (1,))
        
        start = time.time()
        history=model.fit(train_feature, train_label, epochs=100, batch_size=20, validation_data=(validation_feature, validation_label_bin), verbose = 1)
        total_time = time.time()-start
        #print("training time", total_time)
    
    
    
        training_scores = model.evaluate(train_feature,train_label,verbose=1)
        #print ("training loss and accuracy=", training_scores) #to print training accuracy 
    
        prediction = model.predict(validation_feature,verbose = 1)
    
        validation_scores = model.evaluate(validation_feature, validation_label_bin,verbose= 1)
        #print ("validation loss and accuracy=", validation_scores) #to print validation accuracy
        print()
    
    
        return prediction, validation_label, history, total_time, model.evaluate(train_feature,train_label)[1]
    
    ### Dataframes to save fold wise average model performance using training and validation datasets
    training_acc=[]
    validation_acc=[]
    precision=[]
    recall=[]
    specificity = []
    F1_value=[]
    train_time=[]
    MCC_score=[]
    AUPRC = []
    AUROC = []
    
    #%matplotlib inline
    
    if __name__ == "__main__":
        n_folds = 5
        nb_classes=counter
        input_length = features
        
        X = np.array(X,dtype=float)
        Y = np.array(Y, dtype=int)
        
        
        print (X.shape)
        print (Y.shape)
        print ()
    
        ### K-fold Cross-Validation with Training and Validation dataset    
        i = 1
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for train, validation in kfold.split(X, Y):
            print('fold {} is running'.format(i))
            print()
            
            model = None  # Clearing the NN.
            model = create_model(nb_classes, input_length)
            pred, validation_label, history,total_time,train_ac = train_and_evaluate_model(model, X[train], Y[train], X[validation], Y[validation], nb_classes)
            
    
            training_acc.append(train_ac)
    
            ### Calculating actual and predicted class labels for Validation Dataset
            pred_1 = np.argmax(pred, axis=1)
            pred_1 = pred_1.reshape(pred_1.shape+(1,))
    
            Y[validation] = np.array(Y[validation], dtype=int)
            pred_1 = np.array(pred_1, dtype=int)
    
    
            validation_acc.append(accuracy_score(Y[validation], pred_1))
            precision.append(precision_score(Y[validation], pred_1))
            recall.append(recall_score(Y[validation], pred_1))
            tn, fp, fn, tp = confusion_matrix(Y[validation], pred_1).ravel()
            specificity_score = tn / (tn+fp)        
            specificity.append(specificity_score)
            F1_value.append(f1_score(Y[validation], pred_1))
            MCC_score.append(matthews_corrcoef(Y[validation], pred_1))
            train_time.append(total_time)
    
            # Area under the Precision-Recall curve
            positive_class_probs = pred[:, 1]
            positive_class_precision, positive_class_recall, _ = precision_recall_curve(Y[validation], positive_class_probs)
            AUPRC_score = auc(positive_class_recall, positive_class_precision)
            AUPRC.append(AUPRC_score)
    
            # Area under the ROC curve
            AUROC_score = roc_auc_score(Y[validation], positive_class_probs)
            #print ("AUROC score", AUROC)
            AUROC.append(AUROC_score)
        
            i = i + 1    
    
    
    ### Fold Wise results saving
    
    training_data=pd.DataFrame()
    training_data['Fold No.']=range(1,n_folds+1)
    training_data['Training Time (In Seconds)']=train_time
    training_data['Training Accuracy']=training_acc
    training_data['Validation Accuracy']=validation_acc
    training_data['Precision']=precision
    training_data['Recall']=recall
    training_data['Specificity']=specificity
    training_data['F1 Score']=F1_value
    training_data['MCC Score']=MCC_score
    training_data['AUPRC']=AUPRC
    training_data['AUROC']=AUROC
    training_data.to_csv(Result_dir+"/Evaluation metrices_5folds.csv",index=False)
    
    ### Saving the CNN Model        
    model.save(Result_dir+"/DeepRBPL_CNN.h5")
        
if __name__ == "__main__":
    main()