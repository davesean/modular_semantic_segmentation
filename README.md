Based on the [Modular semantic segmentation](https://github.com/ethz-asl/modular_semantic_segmentation) framework from the ETHZ ASL repository. 
Corresponding implementations for the IROS 2018 paper ["Modular Sensor Fusion for Semantic Segmentation"](https://arxiv.org/abs/1807.11249) by Hermann Blum, Abel Gawel, Roland Siegwart and Cesar Cadena.

# Introduction
In this project the idea of quantifying the uncertainty of semantic segmentation networks is explored. This is achieved by utilizing the generative adversarial networks (GAN) ability to generate synthetic images. In this case, a conditional GAN (cGAN) is trained to learn a transformation from semantic segmentations back to RGB images. This can be useful, when the semantic segmentation fails to recognize or misclassifies foreign (Out-of-Distribution) objects. The cGAN then generates a synthetic image based on the semantic segmentation output.

# Semantic Segmentation

# Conditional GAN

# Separate Patch Discriminator

# Evaluation
