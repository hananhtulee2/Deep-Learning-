This Python package was created by
MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
29-Apr-2024 16:55:36

This package contains a TensorFlow model exported from MATLAB.


ISSUES ENCOUNTERED DURING EXPORT FROM MATLAB
--------------------------------------------
Unable to export untrained normalizations for these input layers: 
imageinput. You can train before exporting or set the "Normalization" value 
to "none".

Layer 'scaling': Layer class 'rl.layer.ScalingLayer' was exported into an 
incomplete TensorFlow custom layer file. The custom layer definition must 
be completed or the file must be replaced before the model can be loaded 
into TensorFlow.


USAGE
-----

* To load the model with weights:

    import duancuoiky
    model = duancuoiky.load_model()


* To load the model without weights:

    import duancuoiky
    model = duancuoiky.load_model(load_weights=False)


* To save a loaded model into TensorFlow SavedModel format:

    model.save(<fileName>)


* To save a loaded model into TensorFlow HDF5 format:

    model.save(<fileName>, save_format='h5')


PACKAGE FILES
-------------

model.py
	Defines the untrained model. Modifying this file before calling load_model 
	might cause load_model to give unexpected results.

weights.h5
	Contains the model weights in HDF5 format. Warning: This file is not 
	compatible with the tensorflow.keras.models.load_model() function. See the 
	USAGE section above for instructions on how to load the model and re-save 
	it in standard TensorFlow SavedModel or HDF5 formats.

__init__.py
	Defines the load_model function.
