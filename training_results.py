import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil


def load_histories_appended(histories_path):
    subdirs = ['CNN', 'DCNN1', 'DCNN2', 'DCNN3']

    histories_CNN = pd.DataFrame()
    histories_DCNN1 = pd.DataFrame()
    histories_DCNN2 = pd.DataFrame()
    histories_DCNN3 = pd.DataFrame()

    base_path = histories_path

    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        
        files = os.listdir(subdir_path)
        csv_files = [file for file in files if file.endswith('.csv')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)
            df = pd.read_csv(csv_path)
            
            if subdir == 'CNN':
                histories_CNN = histories_CNN.append(df)
            elif subdir == 'DCNN1':
                histories_DCNN1 = histories_DCNN1.append(df)
            elif subdir == 'DCNN2':
                histories_DCNN2 = histories_DCNN2.append(df)
            elif subdir == 'DCNN3':
                histories_DCNN3 = histories_DCNN3.append(df)
    return histories_CNN, histories_DCNN1, histories_DCNN2, histories_DCNN3



def save_results_plots(output_dir, histories_CNN, histories_DCNN1, histories_DCNN2, histories_DCNN3):

    # Clean the output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define the list of models
    models = ['CNN', 'DCNN1', 'DCNN2', 'DCNN3']

    # Loop over the models
    for i, model in enumerate(models):
        # Create a new figure for each plot
        plt.figure(figsize=(8, 6), dpi=400)

        # Load the training history
        if model == 'CNN':
            history_df = histories_CNN
            title_loss = f'{model} ([8]@ 3x3)'
            title_acc = f'{model} Average ACC per epoch'
        elif model == 'DCNN1':
            history_df = histories_DCNN1
            title_loss = f'{model} ([8,16]@ 3x3)'
            title_acc = f'{model} Average ACC per epoch'
        elif model == 'DCNN2':
            history_df = histories_DCNN2
            title_loss = f'{model} ([8,16,32]@ 3x3)'
            title_acc = f'{model} Average ACC per epoch'
        elif model == 'DCNN3':
            history_df = histories_DCNN3
            title_loss = f'{model} ([8,16,32,64]@ 3x3)'
            title_acc = f'{model} Average ACC per epoch'

        # Calculate the mean training history by grouping the first column
        mean_history_df = history_df.groupby(history_df.columns[0]).mean().reset_index()

        # Plot val_loss vs train
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['loss'], label='train')
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['val_loss'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Mean categorical cross-entropy')
        plt.title(title_loss)
        plt.legend()

        # Save the figure
        filename = os.path.join(output_dir, f'{model}_loss_plot.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Create a new figure for accuracy
        plt.figure(figsize=(8, 6), dpi=400)

        # Plot val_accuracy vs train
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['accuracy'], label='train')
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['val_accuracy'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Mean ACC')
        plt.title(title_acc)
        plt.legend()

        # Save the figure
        filename = os.path.join(output_dir, f'{model}_accuracy_plot.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Create a new figure for precision
        plt.figure(figsize=(8, 6), dpi=400)

        # Plot val_precision vs train
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['precision'], label='train')
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['val_precision'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title(f'{model} Precision')
        plt.legend()

        # Save the figure
        filename = os.path.join(output_dir, f'{model}_precision_plot.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Create a new figure for recall
        plt.figure(figsize=(8, 6), dpi=400)

        # Plot val_recall vs train
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['recall'], label='train')
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['val_recall'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title(f'{model} Recall')
        plt.legend()

        # Save the figure
        filename = os.path.join(output_dir, f'{model}_recall_plot.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    # Create a new figure for Mean AUC Comparison
    plt.figure(figsize=(8, 6), dpi=400)

    # Loop over the models
    for model in models:
        # Load the training history
        if model == 'CNN':
            history_df = histories_CNN
        elif model == 'DCNN1':
            history_df = histories_DCNN1
        elif model == 'DCNN2':
            history_df = histories_DCNN2
        elif model == 'DCNN3':
            history_df = histories_DCNN3

        # Calculate the mean training history by grouping the first column
        mean_history_df = history_df.groupby(history_df.columns[0]).mean().reset_index()

        # Plot mean AUC for each model
        plt.plot(mean_history_df.iloc[:, 0], mean_history_df['val_auc'], label=model)

    # Set labels and title for the Mean AUC Comparison plot
    plt.xlabel('Epoch')
    plt.ylabel('Mean AUC')
    plt.title('Mean AUC Comparison')
    plt.legend()

    # Save the figure
    filename = os.path.join(output_dir, 'mean_auc_plot.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # load and append histories from all folds and repeats of training
    histories_CNN, histories_DCNN1, histories_DCNN2, histories_DCNN3 = load_histories_appended('histories')
    # save plots of the mean of loss, acc and auc for all models
    save_results_plots('train_results', histories_CNN, histories_DCNN1, histories_DCNN2, histories_DCNN3)

    print("Finished succesfully")
