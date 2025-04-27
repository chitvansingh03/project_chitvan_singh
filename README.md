# project_chitvan_singh
Roll No.: 20211072


## PROJECT PROPOSAL – IMAGE AND VIDEO PROCESSING WITH DEEP LEARNING
## NAME: CHITVAN SINGH , ROLL NO: 20211072

### OBJECTIVE
Convolution Neural Networks are becoming a new tool for climate research and forecasting. This is because, they show the promise capture both non linearities and spatial teleconnections that happen in a dynamic fluid like atmosphere and ocean. Here, I will try to leverage, the property of CNNs to understand grids and extract and correlate the features of an image to show how we can use them to forecast a basic atmospheric variable, the 2 meter air temperature normalised anomalised (trend removed). I will be forecasting 2 days from the present using the data present and yesterday (2 days). I will not use air temperature as an input variable, but rather other measured quantities to show that that CNN does extract physics of the system, although to not full extent. Variables include radiation and heat fluxes at surface such as outgoing longwave radiation at the surface (which captures, radiation emitted by earth surface and primarily heats up the atmosphere), downwelling shortwave radiation (sun's radiation), sensible heat flux (heat flux due to convection) and latent heat flux (heat flux ue to moisture condensation/vapourisation). The anomalies of all quantities are used. The aim of this project is to show the applicability of CNNs to use just the radiative and convective data (without any dynamical input) weather forecasting in most basic sense.

Kindly note that model predicts the normalised temperatrue anomalies from the anomalies of various radiation and heat fluxes.

### DATA
I would be using 75 years NCEP NCAR Reanalysis 1 from Physical Sciences Laboratory, NOAA (from 1948 to 2023). The spatial resolution of the data will be 2.50x2.50 and temporal resolution of 1 day (daily mean temperature for e.g.). Training data will be 1948 - 2013 (65 years) and testing will be from 2014 - 2023 (10 years). The climate grid used is from lat = (5N - 35N) and lon = (63E - 93E) at 1.875 degree resolution, which comes to 15 latitude levels and 16 longitude levels. Hence the image is of size 15x16. The channels are added as the variables 2 days (present and yesterday) are stacked hence, for 4 variables and 2 days, we have 8 channels, hence the final image shape becomes 15x16x8. The grid will be centered over central India and forecast will be made over a particular point inside the grid (18.1N, 73.1E which is approximately Pune).

### Preprocessing
The data were in netcdf4 format (.nc). I converted all of them into numpy arrays. Then I computed the normalised anomalies ( (X - daily_mean)/ dail_std_dev), so the values are comparable. The final images were created of shape 15x16x8 during this processing. All of them were stacked along 4th dimesion for easy storage, hence for training - 65 years of daily data we have ~24090 samples and testing , 10 years of daily data ~ 3646 samples. Hence training data is of shape 24090x15x16x8 and testing input data is of shape 3646x15x16x8.

### INPUT
The input for the training will be training data created in pre processing which is a single numpy array of shape (24090, 15, 16, 8) = (no. of samples, no. of rows, no. of columns, no. of channels). For testing, in data folder, 10 samples are kept of shapes (15, 16, 8). these correspond to randomly selected samples starting from 01-01-2014 and every 30th day after that, so one sample from each month is taken.

### OUTPUT
The output will be a scalar value of air temperature anomaly over a 1.8x1.8 degree box over Pune, 2 days from present.

### MODEL
The model contains 1 convolution layers (with ReLU and including max pooling), then 3 fully connected layers with ReLU activation function. One dropout layer would be introduced to reduce the number of parameters. Adam optimizer would be used to optimize the loss function. All the hyper parameters are tested for various values before finalizing them. Kindly note that model predicts the normalised temperatrue anomalies from the anomalies of various radiation and heat fluxes.


## Instructions for The CODE:

* Please ensure that working directory and directory where my files are saved are same, else follow the instructions in interface.py
* I have an example code in "interface.py for you to know the inputs and outputs of the_dataloader, the_trainer, and the_predictor.
* In config.py the variable hyper_param contains all the hyper_paramters, and input_shape contains, the expected shape of a single input image (numpy array).
* Please import all neccesary libraries as in interface.py
* Kindly note the code has been written in python 3.10. Hence please use the same environment.

## Training data:
It's on my drive. Its a 400 MB file, which has test and train data (input and output) are combined. to see it:
use data = np.load() function. x_train = data['x_train'], y_train = data['y_train'] , x_test = data['x_test'], y_test = data['y_test']
* Note, that the code, already has methods to take out x and y test train data.
* Use The following link to download this:
* https://drive.google.com/drive/folders/1Bhggz8ueROvT11u_taYsOpKqkb_-ydZI?usp=sharing

## Important note on results:
Test loss on test data of ~3000 samples from 2014- 2023 was MSE = 0.6973, MAE = 0.66
* The model was not able to predict extreme anomalies |T| > 0.5, well, however it is able to replicate mean behaviour over the time very well. Hence, the anomaies have the correct sign mostly.
* This is illustraed from the follwing plots.

* ./corelation_plot.png

## Kindly contact in case any issue:
EMAIL - chitvan.singh@students.iiserpune.ac.in




