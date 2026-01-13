# MPCI-Bench

This repository contains the source code and benchmark for the paper:

> **MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents**

![MPCI-Bench Case Example](./case_example.png)

---



## VISPR Dataset Dependency
All images used in **MPCI-Bench** are selected from the VISPR dataset:

> Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz.  
> **Towards a Visual Privacy Advisor: Understanding and Predicting Privacy Risks in Images.** > *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017.

An image filtering stage is applied to select privacy-relevant images. The list of required image id is provided in:
`bench/required_image_ids.json`

---

## Dataset Setup
### Step 1: Download VISPR (train split)
* **Images:** [train2017.tar.gz](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017.tar.gz)
* **Annotations:** [train2017_anno.tar.gz](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017_anno.tar.gz)

### Step 2: Filter Files
After extraction, Filter the images according to the IDs provided in `bench/required_image_ids.json`. Place the filtered images and their corresponding annotations into:
`bench/vispr/train2017/`

We will relase the data construction code soon.
