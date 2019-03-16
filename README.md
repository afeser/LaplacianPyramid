# LaplacianPyramid

## GPU Implementation of filters
The library includes GPU implementation of filters and laplacian pyramids. Also Local Laplacian Pyramid and blend example in openCV implemented. GPU code runs very quickly compared with the original MATLAB code given here [LLP](https://people.csail.mit.edu/sparis/publi/2011/siggraph/). Unfortunately this my first project in my life, so codes are not well design, in fact they have a lot of bugs in it! 


## Files
LaplacianPyramid.cu -> Some functions on images, creating Gaussian and Gaplacian images, openCV example of blend image, sharpen etc.
LocalLaplacianPyramid.cu -> Local Laplacian Pyramid CUDA implementation
main.cu -> A simple interface to use the above
eskiKodlar -> Tried, but not used, like recyle bin

## Some important notes

Library depends on MyLibrary library(I know it is so stupid), which can be found [here](https://github.com/afeser/MyLibrary)

