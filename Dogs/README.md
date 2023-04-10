# Dog_Breed
dog breed prediction app

This app uploads a picture of a dog or human, determines if it is a dog or a human then tells what breed it looks like

To run this program, navigate to the capstone folder containing the manage.py folder, then type the command python manage.py runserver.
Next put this web address into your web browser http://127.0.0.1:8000/.  The web app should be running at that address.


Project Definition:

This project uses a model to first determine whether an image is of a dog or a human, then determines what breed of dog the human or dog looks most like.

The this is an quick and easy way to find out a dog breed from and image using a deep learning model.

I used the accuracy metric to determine the quality of the model.

Analysis:

Minimal preprocessing was required because the training images were provided in the dog breed jupyter notebook.

I used a couple of different transfer learned models then did some additional training.  Due to some logistal issues and python version changes the model that would work in the web app was VGG-16.  It was not the best performing model but it could handle the data format and version changes.

For data visualization I displayed a few of the training images in the notebook.

Methodology:

No data pre-processing was necessary because the image data was provided in the notebook, and it is generally not necessary to pre-processing image data.

I used transfer learning from various pre-trained deep learning models to classify the dog breeds.  I tried various sizes of dense layers to refine the classification accuracy.

I was not able to retrain my model within the web app because I was not able to download the training or test data from the jupyter notebook.  I was unable to train my models locally because the data could not be found in project git hub or the jupyter notebook.

