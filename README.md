# DDMP-Project 
Project: **Speech representations for predicting listeners perception** \
Report on the tasks carried out between the 19th of February and the 13th of March

# Cluster connection
Thanks to the authorization given by the university, we were able to set the VPN and connect to the cluster both from the machines in the classroom and our laptops.

# Data access
The CPC data was available in the cluster so we were able to access it without problems.

# New Backbone and Feature Extraction
For this project we use a new backbone: [w2v-BERT2.0](https://huggingface.co/facebook/w2v-bert-2.0) for speech embedding extraction.
![Example Image](https://github.com/shubh2001/DDMP-Project/blob/main/model.png)
|:--:| 
| *w2v-bert-2.0* |

For this model we create a feature extraction [script](https://github.com/shubh2001/DDMP-Project/blob/main/main.py) which is based on @tiagoCuervo 's script [link](https://github.com/tiagoCuervo/fairseq/blob/main/examples/wav2itl/scripts/hubert_get_feats.py). Following changes were made:
* We use **torchaudio** instead of soundfile.
* We resample all read audio files to **16,000 Hz** to align with the new backbone.

Feature extraction was done on cluster with an interactive GPU environment and took roughly 50 minutes for a single train.x file. (x = {1,2,3})

# Creating a virtual work environment
We created a YAML file to have the same setup for our conda virtual environment. We use this virtual environment to set the sequence modeling toolkit, fairseq, that we use to run our training.                                                                                

# Furhter issues
The task class of CPC2 was defined in two different locations. This set us back for some time as we were figuring out the source of the following error.
ValueError: Cannot register duplicate task (cpc2).
After figuring out the solution, only some minor errors arose which were then solved. We are now able to run the training on the cluster.

# Figures on the training currently running
![Example Image](https://github.com/shubh2001/DDMP-Project/blob/main/Training_%231.png)
|:--:| 
| *Training Results* |
\
![Example Image](https://github.com/shubh2001/DDMP-Project/blob/main/validatation_%231.png)
|:--:| 
| *Validation Results* |
# Conlclusion
Based on the results of this training, the model with the best performance is XXX

# Further steps
Now, we are working on tuning the hyperparameters of our model to further improve the performance of the XXX model. 
