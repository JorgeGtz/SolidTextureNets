# SolidTextureNets

PyTorch implementation of the solid texture synthesis model in the journal article [**On Demand Solid Texture Synthesis Using Deep 3D Networks**](https://hal.archives-ouvertes.fr/hal-01678122v3) published in Computer Graphics Forum. [DOI:10.1111/cgf.13889](https://doi.org/10.1111/cgf.13889)

Code based on Gatys' Neural Style Transfer [implementation](https://github.com/leongatys/PytorchNeuralStyleTransfer)

## Training

The python script **train_slice.py** trains a generator network and produces the file ***params.pytorch** that contains the trained parameters.

It requires the libraries: PIL and PyTorch.

The VGG-19 perceptual loss between 2D images uses Gatys' implementation mentioned above.

To run the code you need to get the pytorch VGG19-Model from the bethge lab by running (script from https://github.com/leongatys/PytorchNeuralStyleTransfer) :
```
sh download_models.sh 
```
Using [display](https://github.com/szym/display) is optional.

The names of the example textures and the associated training directions are defined by the lists **input_names** and **directions**.

The example textures go in the **Textures** folder.

## Sampling on-demand

## Visualization
