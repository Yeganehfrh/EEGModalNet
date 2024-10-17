# open model
from ...EEGModalNet import Classifier
from ...EEGModalNet import load_data
import numpy as np
import torch
from torch.autograd import grad
import pandas as pd


# Integrated Gradients function
def integrated_gradients(model, inputs, target_class, baseline=None, steps=50):
    if baseline is None:
        # Set the baseline to zero (same shape as input)
        baseline = torch.zeros_like(inputs)

    # Scale inputs and compute gradients at each interpolation point
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).flatten(0, 1).requires_grad_(True)

    # Initialize attributions
    total_gradients = torch.zeros_like(inputs)

    for scaled_input in scaled_inputs:
        # Perform forward pass
        output = model(scaled_input.unsqueeze(0))  # Add batch dimension
        # Extract the probability for the target class
        if target_class == 1:
            target = output  # Use the probability of class 1
        elif target_class == 0:
            target = 1 - output  # Use the probability of class 0

        # Compute gradients
        grads = grad(target, scaled_input, retain_graph=True)[0]
        total_gradients += grads

    # Average gradients over all steps
    avg_gradients = total_gradients / steps

    # Compute the attributions
    attributions = (inputs - baseline) * avg_gradients
    return attributions.detach().cpu().numpy()


if __name__ == '__main__':

    eeg_data_path = 'data/OTKA/experiment_EEG_data.nc5'
    session_data_path = 'data/OTKA/behavioral_data.csv'
    channels = ['Oz', 'Fz', 'Cz', 'Pz', 'Fp1', 'Fp2', 'F1', 'F2']
    time_dim = 512
    n_splits = 3
    n_subject = 52
    dropout_rate = 0.4
    model_name = 'label_classifier17092024'
    X_input, y, train_val_splits, channels = load_data(eeg_data_path, session_data_path, channels, time_dim=time_dim,
                                                       n_subject=n_subject, n_splits=n_splits)

    all_attributions = {}
    model_names = ['label_classifier17092024_fold-1_best_val_acc.model.keras', 'label_classifier17092024_fold-2_best_val_acc.model.keras']
    # Load the model
    for i in range(2):
        print(f'>>>>>> Load Model {i+1}')
        tmp = []
        val_idx = train_val_splits[i][1]
        model = Classifier(feature_dim=len(channels), dropout_rate=dropout_rate)
        model.load_weights(f'logs/{model_names[i]}')

        for sample_idx in range(X_input[val_idx].flatten(0, 1).shape[0]):
            if sample_idx % 100 == 0:
                print(f'>>>>>> Sample {sample_idx}')
            eeg_sample = X_input[val_idx].flatten(0, 1)[sample_idx].float().unsqueeze(0)
            eeg_label = y[val_idx].flatten(0, 1)[sample_idx].item()

            # Call Integrated Gradients function
            tmp.append(integrated_gradients(model, eeg_sample, target_class=eeg_label))

        all_attributions[f'model_{i}'] = np.array(tmp)

    # Save the all_attributions
    all_attributions = np.array(list(all_attributions.values()))
    np.save('logs/all_attributions.npy', all_attributions)
