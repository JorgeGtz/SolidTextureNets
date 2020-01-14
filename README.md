# SolidTextureNets

PyTorch implementation of the solid texture synthesis model in the journal article [**On Demand Solid Texture Synthesis Using Deep 3D Networks**](https://hal.archives-ouvertes.fr/hal-01678122v3) published in Computer Graphics Forum. [DOI:10.1111/cgf.13889](https://doi.org/10.1111/cgf.13889)

Code based on Gatys' Neural Style Transfer [implementation](https://github.com/leongatys/PytorchNeuralStyleTransfer)

## Training

The python script `train_slice.py` trains a generator network and produces the file `params.pytorch` that contains the trained parameters.

It requires the libraries: PIL and PyTorch.

The VGG-19 perceptual loss between 2D images uses Gatys' implementation mentioned above.

To run the code you need to get the pytorch VGG19-Model from the bethge lab using the script `download_models.sh` from https://github.com/leongatys/PytorchNeuralStyleTransfer


Using [display](https://github.com/szym/display) is optional.

The names of the example textures and the associated training directions are defined by the lists **input_names** and **directions**.

The example textures go in the **Textures** folder.

## Sampling on-demand

The python script `sample_on_demand.py` loads the trained parameters and synthesizes a block of texture of sizes **total_H**, **total_W**, **total_D** formed of blocks of sizes **piece_height**, **piece_width**, **piece_depth**.

It requires the libraries: PIL, PyTorch and cupy

Indicate the location of the trained model in **model_folder**.

The output file `*.npy` is a numpy 4D array with the number of channels (BGR) in the first dimension and the spatial dimensions in the next three dimensions. 

## Visualization

### ParaView
One visualization option is [ParaView](https://www.paraview.org/). It requires to convert the `*.npy` file to VTK. We used the library [PyEVTK](https://bitbucket.org/pauloh/pyevtk/src/default/) (see **gridToVTK**).

### OpenGL
Additionally we provide a PyOpenGL script `render_cube.py` that loads the `*.npy` file and uses it to render a simple cube. 
The code requires OpenGL and the python libraries: pyopengl, sdl2 and pyglm 
**texture** defines the path to the texture (*.npy) to apply.
Use the key 's' to stop and start the rotation of cube

