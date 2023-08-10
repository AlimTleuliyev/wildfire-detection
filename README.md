# Wildfire Detection Research

Welcome to the Wildfire Detection Research repository, focusing on fire detection using computer vision. This project aims to contribute to wildfire prevention efforts by leveraging machine learning to detect fire and smoke instances in images.

<p align="center">
  <img src="src/intro-gif.gif" alt="gif" width = 600>
</p>

---

## Dataset

We utilized the [D-Fire dataset](https://github.com/gaiasd/DFireDataset), a curated collection of 21,000 labeled images, each annotated in YOLO format. The dataset focuses on fire and smoke instances, while also encompassing diverse visual cues, including non-fire images that resemble fire-like patterns.

We have explored many different datasets. Here is the summary:

| Dataset                                                                                                                                                                                    | Image type      | Image view                           | \# images                                                                                                                                                                                                                                          | \# classes                                                                                                     | Bboxes |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------ |
| [Wildfire Detection Image Data](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data)                                                                                   | RGB             | Regular                              | 1800 train and val, 75 test                                                                                                                                                                                                                        | fire, no fire                                                                                                  | no     |
| [Fire Detection Using Surveillance Camera on Roads](https://www.kaggle.com/datasets/tharakan684/urecamain)                                                                                 | RGB             | Surveillance Camera                  | 10000 images                                                                                                                                                                                                                                       | fire, no fire                                                                                                  | no     |
| [FIRESENSE](https://www.kaggle.com/datasets/chrisfilo/firesense) (Videos)                                                                                                                  | RGB             | Regular, Surveillance Camera         | a) for flame detection 11 positive and 16 negative videos are provided, while<br>b) for smoke detection, 13 positive and 9 negative videos are provided.                                                                                           | fire, no fire, smoke, no smoke                                                                                 | no     |
| [Aerial Rescue Object Detection](https://www.kaggle.com/datasets/julienmeine/rescue-object-detection)                                                                                      | RGB             | Regular, drone                       | 29810 images                                                                                                                                                                                                                                       | human, fire, vehicle                                                                                           | yes    |
| [Fire detection dataset](https://www.kaggle.com/datasets/jimishpatel/fire-detection-dataset)                                                                                               | RGB             | Regular, Surveillance Camera         | 3894 images                                                                                                                                                                                                                                        | fire, no fire                                                                                                  | no     |
| [Forest Fire](https://www.kaggle.com/datasets/kutaykutlu/forest-fire)                                                                                                                      | RGB, Gray Scale | Regular, Drone, Surveillance Camera  | 15800 images                                                                                                                                                                                                                                       | fire, smoke                                                                                                    | no     |
| [Forest Fire Images](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images)                                                                                                  | RGB             | Regular                              | 5000 images                                                                                                                                                                                                                                        | fire, no fire                                                                                                  | no     |
| [Fire Detection in YOLO format](https://www.kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format)                                                                                   | RGB             | Regular                              | 500 images                                                                                                                                                                                                                                         | fire                                                                                                           | yes    |
| [FLAME 2: FIRE DETECTION AND MODELING: AERIAL MULTI-SPECTRAL IMAGE DATASET](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) | RGB, IR         | Drone                                | 53451 RGB, 53451 IR                                                                                                                                                                                                                                | fire and smoke, fire and no smoke, no fire and smoke, no fire and no smoke                                     | no     |
| [Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset)                                                                                                          | RGB             | Regular                              | 1900 images                                                                                                                                                                                                                                        | fire, no fire                                                                                                  | no     |
| [Open Wildfire Smoke Datasets](https://github.com/aiformankind/wildfire-smoke-dataset/tree/master)                                                                                         | RGB             | Surveillance Camera                  | 2192 images                                                                                                                                                                                                                                        | smoke                                                                                                          | yes    |
| [AIDER: Aerial Image Database for Emergency Response applications](https://github.com/ckyrkou/AIDER/tree/master)                                                                           | RGB             | Aerial view, regular                 | 500 images for each disaster class and over 4000 images for the normal class.                                                                                                                                                                      | Fire/Smoke, Flood, Collapsed Building/Rubble, and Traffic Accidents, as well as one class for the Normal case. | no     |
| [Furg Fire Dataset](https://github.com/steffensbola/furg-fire-dataset)                                                                                                                     | RGB             | Regular                              | 21 videos                                                                                                                                                                                                                                          | fire                                                                                                           | yes    |
| [Mivia Fire Detection](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)                                                                                    | RGB             | Regular                              | 14 videos characterized by the presence of the fire and the last 17 videos which do not contain any event of interest                                                                                                                              | fire, no fire                                                                                                  | no     |
| [FireNet](https://github.com/OlafenwaMoses/FireNET)                                                                                                                                        | RGB             | Regular                              | 500 images                                                                                                                                                                                                                                         | fire                                                                                                           | yes    |
| [FIRE Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)                                                                                                                   | RGB             | Regular                              | fire_images folder contains 755 outdoor-fire images some of them contains heavy smoke, the other one is non-fire_images which contain 244 nature images (eg: forest, tree, grass, river, people, foggy forest, lake, animal, road, and waterfall). | fire, no fire                                                                                                  | no     |
| [Fire Detection v2](https://universe.roboflow.com/yi-shing-group-limited/fire-detection-v2-yn3wz)                                                                                          | RGB             | Regular                              | 600 images                                                                                                                                                                                                                                         | scale1fire, scale2fire, scale3fire                                                                             | yes    |
| [fireDetection Computer Vision Project](https://universe.roboflow.com/school-tvtyg/firedetection-xxwxc)                                                                                    | RGB             | Regular                              | 9681 images                                                                                                                                                                                                                                        | Fire, fire                                                                                                     | yes    |
| [D-Fire](https://github.com/gaiasd/DFireDataset)                                                                                                                                           | RGB             | Regular, Aerial, Surveillance Camera | 21000 images                                                                                                                                                                                                                                       | Fire, Smoke                                                                                                    | yes    |
| [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset)                                                                                                                    | RGB             | Regular                              | 3000 images                                                                                                                                                                                                                                        | Fire, Smoke, Neutral                                                                                           | no                                                                                                                                                                                                               |

---

## Model Training

We trained the YOLOv8 model by [Ultralytics](https://github.com/ultralytics/ultralytics) on the D-Fire dataset to achieve accurate fire and smoke detection. Our research not only focuses on achieving high accuracy but also on optimizing model parameters and hyperparameters to ensure efficiency and speed.

---

## Results

The trained YOLOv8 model demonstrated impressive performance on the D-Fire test dataset, with mAP@50 scores and inference time across different model sizes as follows. Evaluation was done using NVIDIA A100-SXM4-40. Resolution of input images was 640x640.

| Model Size  | mAP@50 | Inference (ms) |
|-------------|--------|----------------|
| Nano        | 0.787  |     0.422      |
| Small       | 0.798  |     0.773      |
| Medium      | 0.801  |     1.532      |
| Large       | 0.812  |     2.342      |
| Extra Large | 0.814  |     3.465      |

For a detailed exploration of our training process and insights, we invite you to read our comprehensive guide on Medium: [Guide Link](https://medium.com/your-article-link)

---

## Future Implications

This research underscores the potential of computer vision in addressing real-world challenges, such as wildfire detection. As technology evolves, integrating machine learning tools into wildfire prevention and emergency response strategies could significantly enhance our ability to detect and mitigate wildfires effectively.

---

## Streamlit App

For a practical demonstration of our research, you can interact with our Wildfire Detection App, powered by the YOLOv8 model. This app allows you to upload images and observe the model's detection capabilities in action. To experience the app, visit: [Wildfire Detection App](https://wildfire-detection.streamlit.app)

---

## Disclaimer

Please note that while our Streamlit app demonstrates the capabilities of our model, it is intended for demonstration purposes and may not be suitable for critical wildfire detection applications.

---

## License

This project is licensed under the [MIT License](LICENSE).
