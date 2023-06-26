import os
import pandas as pd
import csv
from statistics import mean, stdev

def training_results(histories_path, mean_history_path, mean_checkpoints_path):
    df = list()
    for cv in os.listdir(histories_path):
            df.append(pd.read_csv(histories_path + cv))
    cont = 1
    f = open(mean_history_path, 'w')
    f1 = open(mean_checkpoints_path, 'w')
    epochs = 100 #hardcoded
    columns = ["Epochs", "loss", "std", "accuracy", "std", "precision", "std", "recall", "std", "auc", "std",
                "val_loss", "std", "val_accuracy", "std", "val_precision", "std", "val_recall", "std", "val_auc", "std"]
    writer = csv.writer(f)
    writer.writerow(columns)
    writer1 = csv.writer(f1)
    writer1.writerow(columns)
    for i in range(epochs):
        train_loss, train_accuracy, train_precision, train_recall, train_auc = list(), list(), list(), list(), list()
        val_loss, val_accuracy, val_precision, val_recall, val_auc = list(), list(), list(), list(), list()
        for d in df:
            train_loss.append(d.iloc[i]['loss'])
            train_accuracy.append(d.iloc[i]['accuracy'])
            train_precision.append(d.iloc[i]['precision'])
            train_recall.append(d.iloc[i]['recall'])
            train_auc.append(d.iloc[i]['auc'])
            val_loss.append(d.iloc[i]['val_loss'])
            val_accuracy.append(d.iloc[i]['val_accuracy'])
            val_precision.append(d.iloc[i]['val_precision'])
            val_recall.append(d.iloc[i]['val_recall'])
            val_auc.append(d.iloc[i]['val_auc'])
        mean_loss = mean(train_loss)
        mean_accuracy = mean(train_accuracy)
        mean_precision = mean(train_precision)
        mean_recall = mean(train_recall)
        mean_auc = mean(train_auc)
        mean_val_loss = mean(val_loss)
        mean_val_accuracy = mean(val_accuracy)
        mean_val_precision = mean(val_precision)
        mean_val_recall = mean(val_recall)
        mean_val_auc = mean(val_auc)
        std_loss = stdev(train_loss)
        std_accuracy = stdev(train_accuracy)
        std_precision = stdev(train_precision)
        std_recall = stdev(train_recall)
        std_auc = stdev(train_auc)
        std_val_loss = stdev(val_loss)
        std_val_accuracy = stdev(val_accuracy)
        std_val_precision = stdev(val_precision)
        std_val_recall = stdev(val_recall)
        std_val_auc = stdev(val_auc)

        r = [str(cont), str(mean_loss), str(std_loss), str(mean_accuracy), str(std_accuracy), str(mean_precision), str(std_precision),
             str(mean_recall), str(std_recall), str(mean_auc), str(std_auc), 
             str(mean_val_loss), str(std_val_loss), str(mean_val_accuracy), str(std_val_accuracy), str(mean_val_precision), str(std_val_precision), 
             str(mean_val_recall), str(std_val_recall), str(mean_val_auc), str(std_val_auc)]
        writer.writerow(r)
        if cont % 10 == 0:
            writer1.writerow(r)
        cont += 1
    f.close()
    f1.close()

def get_higher_metric_model_id(path, metric):
    df = pd.read_csv(path)
    return int(df.iloc[df.idxmax()[metric], :]["Epochs"])

def get_statistic_pivot(histories_path, model_epoch, metric):
    # df = pd.read_csv(path)
    df=[]
    for cv in os.listdir(histories_path):
        df.append(pd.read_csv(histories_path + cv))
    
    return int(df.iloc[model_epoch, :]["Epochs"])

if __name__ == "__main__":
    CNN_histories_path = 'histories/CNN/'
    CNN_mean_history_path = 'results/CNN_mean_history.csv'
    CNN_mean_checkpoints_path = 'results/CNN_mean_checkpoints_history.csv'

    DCNN1_histories_path = 'histories/DCNN1/'
    DCNN1_mean_history_path = 'results/DCNN1_mean_history.csv'
    DCNN1_mean_checkpoints_path = 'results/DCNN1_mean_checkpoints_history.csv'

    DCNN2_histories_path = 'histories/DCNN2/'
    DCNN2_mean_history_path = 'results/DCNN2_mean_history.csv'
    DCNN2_mean_checkpoints_path = 'results/DCNN2_mean_checkpoints_history.csv'

    DCNN3_histories_path = 'histories/DCNN3/'
    DCNN3_mean_history_path = 'results/DCNN3_mean_history.csv'
    DCNN3_mean_checkpoints_path = 'results/DCNN3_mean_checkpoints_history.csv'

    training_results(CNN_histories_path, CNN_mean_history_path, CNN_mean_checkpoints_path)
    CNN_pivot = get_higher_metric_model_id(CNN_mean_checkpoints_path, "val_auc")
    print('The best CNN model performance was in epoch: ', CNN_pivot)

    training_results(DCNN1_histories_path, DCNN1_mean_history_path, DCNN1_mean_checkpoints_path)
    DCNN1_pivot = get_higher_metric_model_id(DCNN1_mean_checkpoints_path, "val_auc")
    print('The best DCNN1 model performance was in epoch: ', DCNN1_pivot)

    training_results(DCNN2_histories_path, DCNN2_mean_history_path, DCNN2_mean_checkpoints_path)
    DCNN2_pivot = get_higher_metric_model_id(DCNN2_mean_checkpoints_path, "val_auc")
    print('The best DCNN2 model performance was in epoch: ', DCNN2_pivot)

    training_results(DCNN3_histories_path, DCNN3_mean_history_path, DCNN3_mean_checkpoints_path)
    DCNN3_pivot = get_higher_metric_model_id(DCNN3_mean_checkpoints_path, "val_auc")
    print('The best DCNN3 model performance was in epoch: ', DCNN3_pivot)

