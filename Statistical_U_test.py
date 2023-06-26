import os
from pathlib import Path
from statistics import mean
from scipy.stats import mannwhitneyu
import pandas as pd
import csv

def check_path(filename, allowed_extensions):
    if allowed_extensions in filename:
        return True

def vector_extraction(pivot_id, model_history_path, epochs):
    df, pivot_model, ramining_models = list(), list(), list()
    cont = 1
    for csv_file in os.listdir(model_history_path):
        if (check_path(csv_file, '.csv')):
            df.append(pd.read_csv(model_history_path + csv_file))   
    
    for i in range(epochs):
        aux = []
        if cont % 10 == 0:
            for d in df:
                if pivot_id == cont:
                    pivot_model.append(d.iloc[i]['val_auc'])
                else:
                    aux.append(d.iloc[i]['val_auc'])
            if pivot_id != cont:
                ramining_models.append(aux)
        cont += 1
    return pivot_model, ramining_models

def mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq):
    output_file = open(Statistic_result_path, 'w')
    columns = ["Pivot epoch", "Other model epoch", "p-value"]
    writer = csv.writer(output_file)
    writer.writerow(columns)
    for i, y in enumerate(ramining_models, start=1):
        if i*check_freq >= pivot_id:
            i = i + 1
        U1, p_value = mannwhitneyu(pivot_model, y, alternative="two-sided", method="asymptotic")
        print(f'P value for {int(pivot_id)} vs {i*check_freq} is: {p_value}')
        values = [str(pivot_id), str(i*check_freq), str(p_value)]
        writer.writerow(values)

    

if __name__ == "__main__":
    epochs=100
    check_freq=20
    print("------- Statistical test for the CNN model ----------")
    model_history_path = 'histories/CNN/'
    model_checkpoints_path = 'results/CNN_mean_checkpoints_history.csv'
    Statistic_result_path = 'results/CNN_statistic_results.csv'
    data = pd.read_csv(model_checkpoints_path)
    pivot_id = data.iloc[data.idxmax()["val_auc"], :]["Epochs"]
    pivot_model, ramining_models = vector_extraction(pivot_id, model_history_path, epochs)
    mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq)

    print("------- Statistical test for the DCNN1 model ----------")
    model_history_path = 'histories/DCNN1/'
    model_checkpoints_path = 'results/DCNN1_mean_checkpoints_history.csv'
    Statistic_result_path = 'results/DCNN1_statistic_results.csv'
    data = pd.read_csv(model_checkpoints_path)
    pivot_id = data.iloc[data.idxmax()["val_auc"], :]["Epochs"]
    pivot_model, ramining_models = vector_extraction(pivot_id, model_history_path, epochs)
    mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq)

    print("------- Statistical test for the DCNN2 model ----------")
    model_history_path = 'histories/DCNN2/'
    model_checkpoints_path = 'results/DCNN2_mean_checkpoints_history.csv'
    Statistic_result_path = 'results/DCNN2_statistic_results.csv'
    data = pd.read_csv(model_checkpoints_path)
    pivot_id = data.iloc[data.idxmax()["val_auc"], :]["Epochs"]
    pivot_model, ramining_models = vector_extraction(pivot_id, model_history_path, epochs)
    mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq)

    print("------- Statistical test for the DCNN3 model ----------")
    model_history_path = 'histories/DCNN3/'
    model_checkpoints_path = 'results/DCNN3_mean_checkpoints_history.csv'
    Statistic_result_path = 'results/DCNN3_statistic_results.csv'
    data = pd.read_csv(model_checkpoints_path)
    pivot_id = data.iloc[data.idxmax()["val_auc"], :]["Epochs"]
    pivot_model, ramining_models = vector_extraction(pivot_id, model_history_path, epochs)
    mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq)