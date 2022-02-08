# PM-Net
A siamese network based on CNN for match feature point of two images.

You can achieve training data from [link](http://phototour.cs.washington.edu/patches/default.htm). It is a dataset for image match.

`xy_surf_descriptor.py` is work for getting pixel coordinate of feature point.

`Get_data_np.py` is work for save patch of images as numpy array.

`Model.py` is work for building and training of PM-Net.

`Test.py` is work for model test.

`App_v1.py` is work for predicting the column of pixel coordinate and calculating the accuracy. 
