# LaCATHODE

Code base for the paper [Resonant anomaly detection without background sculpting](https://arxiv.org/abs/2210.14924).

LatentCATHODE (LaCATHODE) is a new anomaly detection algorithm that tightly follows the [CATHODE protocol](https://arxiv.org/abs/2109.00546) but moves the classification task into the latent space of the normalizing flow. Since the flow has been trained to map sideband events to a unit normal distribution for every conditional mass value, the input to the classifier is decorrelated from the mass. Evaluating the signal region–trained classifier on the sideband region thus does not lead to a sculpted background distribution as is observed in other weakly supervised approaches.

To see the definition of signal region and sideband region please see: [SB-SR](SB-SR.pdf)

## Citation
If you use **LaCATHODE** for your research, please cite:  
- *"Resonant anomaly detection without background sculpting"*,  
By Anna Hallin, Gregor Kasieczka, Tobias Quadfasel, David Shih, and Manuel Sommerhalder. <br>
[Phys. Rev. D 107, 114012](https://doi.org/10.1103/PhysRevD.107.114012). 

## Reproducing the paper results

As LaCATHODE is simply a modification of CATHODE where the classification task is moved into the latent space, this branch of the main CATHODE repo simply modifies the CATHODE mode as such and adds more evaluation functions to test it along CATHODE and an idealized anomaly detector for background sculpting. Note that the latter two are partially trained using the `main` CATHODE branch of this repo.

Instructions on how to download and preprocess the data is shown in `run_data_preparation_latentCATHODE.sh`.

Instructions on the trainings of LaCATHODE, CATHODE and the idealized anomaly detector are provided in `run_trainings_latentCATHODE.sh`.

Finally, the study of how each method sculpts background and how different data representations affect the significance improvement can be found in `bkg_sculpting_study.ipynb`.
