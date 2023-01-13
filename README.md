# Physical Super Resolution (SR) Against Diffraction Blur

#### Qian Huang
#### 1/12/2023

**Demo notebook**: demo.ipynb

**Motivation**: Fisher information suggests we can go beyond diffraction limit given prior knowledge, which can be effectively learned by neural algorithm on large datasets.

**Goal**: develop an neural algorithm to restore images impacted by diffraction blur.

**Method**:
* Develop a SR neural algorithm that can use SRCNN or EDSR, two classic SR networks, as its backbone for physical SR.
* Simulate the camera and develop an associate forward model to generate degraded images from high quality image datasets like DIV2K
* Use generated samples to train the network.
* Save the network weights that perform best on validation set and use them for future inference.
