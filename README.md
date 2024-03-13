# DDMP-Project 
Report on the tasks carried out between the 19th of February and the 13th of March

# Cluster connection
Thanks to the authorization given by the university, we were able to set the VPN and collect to the cluster both from the machines in the classroom and our laptops.

# Data access
The data was available in the cluster so we were able to access it without problems

# Code restructuration (Shubh)
The issues described by Shubh

# Creating a virtual work environment
We created a YAML file to have the same setup for our conda virtual environment. We use this virtual environment to set the sequence modeling toolkit, fairseq, that we use to run our training.

# Furhter issues
The task class of CPC2 was defined in two different locations. This set us back for some time as we were figuring out the source of the following error.
ValueError: Cannot register duplicate task (cpc2).
After figuring out the solution, only some minor errors arose which were then solved. We are now able to run the training on the cluster.

# Figures on the training currently running

# Conlclusion
Based on the results of this training, the model with the best performance is XXX

# Further steps
Now, we are working on tuning the hyperparameters of our model to further improve the performance of the XXX model. 
