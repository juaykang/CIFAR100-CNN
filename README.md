This project implements a LeNet-5-style Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-100 dataset. 

## Objectives
- Implement the LeNet-5 CNN architecture from scratch using `torch.nn.Module`.
- Count the number of trainable parameters in the model.
- Train and evaluate the model under various configurations (batch size, learning rate, epochs).
- Analyze training performance through accuracy and loss across different hyperparameters.

## Model Architecture
The LeNet-5 model includes:
- Two convolutional layers with ReLU activation and MaxPooling
- Three fully connected layers
- Intermediate tensor shape tracking for each major layer

## Files Included
- `student_code.py`: Custom LeNet model and training loop
- `train_cifar100.py`: Script to train the CNN on CIFAR-100
- `eval_cifar100.py`: Evaluate the trained model on the validation set
- `dataloader.py`: CIFAR-100 dataset loader and transformation utilities
