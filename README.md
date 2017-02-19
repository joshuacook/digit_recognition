# Content: Deep Learning
## Project: Build a Digit Recognition Program

### Table of Contents
1. [Project Overview](#project-overview)

<a name="project-overview"></a>
## Project Overview

In this project, you will use what you've learned about deep neural networks and 
convolutional neural networks to create a program that prints numbers it observes 
in real time from images it is given. 

To implement your program, you will:

1. design and test a model architecture that can identify sequences of digits in an image
1. train that model so it can decode sequences of digits from natural images by using the [Street View House Numbers (SVHN) dataset][SVHN]
1. test your model using your program on newly-captured images
1. upon obtaining meaningful results, refine your implementation to also *localize where numbers are on the image*, and test this localization on newly-captured images
   
As an optional bonus to this project, you can integrate your work into a live camera application. 

<a name="submission-requirements"></a>
## Submission Requirements

You will be required to submit three components as part of a completed project:

1. a library of code dedicated to the construction of your digit recognition system
1. an jupyter notebook from which you run this code
1. a report detailing the analysis of your implementation and results

<a name="software-requirements"></a>
## Software Requirements

This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html) (v0.17)
- [TensorFlow](http://tensorflow.org)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

In addition to the above, for those optionally seeking to use image processing software, you may need one of the following:

- [PyGame](http://pygame.org/)
- [OpenCV](http://opencv.org/)

For those optionally seeking to deploy an Android application:

- Android SDK & NDK (see this [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md))

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

<a name="optional-installs"></a>
### Optional Installs

`pygame` and `OpenCV` can then be installed using one of the following commands:

**opencv**  
`conda install -c menpo opencv=2.4.11`

**PyGame:**  
Mac:  `conda install -c https://conda.anaconda.org/quasiben pygame`  
Windows: `conda install -c https://conda.anaconda.org/tlatorre pygame`  
Linux:  `conda install -c https://conda.anaconda.org/prkrekel pygame`

Helpful links for installing and using PyGame:

   - [Getting Started](https://www.pygame.org/wiki/GettingStarted)
   - [PyGame Information](http://www.pygame.org/wiki/info)
   - [Google Group](https://groups.google.com/forum/#!forum/pygame-mirror-on-google-groups)
   - [PyGame subreddit](https://www.reddit.com/r/pygame/)



[SVHN]: http://ufldl.stanford.edu/housenumbers/


