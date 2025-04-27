import sys
sys.path.append(r'E:\Academics_IISER\Sem8\IV_Process\notebook1')

# replace MyCustomModel with the name of your model
from model import CNNModel as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import train_model as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import predictor_airtemp_anomaly as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import UnicornImgDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import unicornLoader as the_dataloader

from config import hyper_param, input_shape


#Example of running the model

## Add the paths

#Note : weight_path - path to final training weights, data_path = path to 1 of the 10 test image
#path in the function the_dataloader = path to training data.

weights_path = "E:\\Academics_IISER\\Sem8\\IV_Process\\notebook1\\checkpoints\\cnn_weights.pth"
data_path = "E:\Academics_IISER\\Sem8\\IV_Process\\notebook1\\data\\input_output_0.npz"

# this gives , train_loader and val_loader which are very important for training. 
# It also gives train_dataset and val_dataset which can be used to understand training data.

## Dataloader
(train_loader, val_loader), (train_dataset, val_dataset) = the_dataloader(
    "E:\\Academics_IISER\\Sem8\\IV_Process\\test_train_data\\train_test_data.npz")

## Training loop
the_trainer(input_shape= input_shape, model = TheModel,
            hyper_param = hyper_param, 
            train_loader = train_loader, val_loader = val_loader)

## Prediction - Assumes in general one numpy image at a time.
the_predictor(TheModel, weights_path, data_path, hyper_param)

## Note of the above commands are not given to above functions they will not work


