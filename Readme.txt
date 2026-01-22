The following intermediate codes/datasets are needed:
triinv.m
iccorrelate.m
uniformer.m
Figure7_HookerCase_BreiGCMRGKnock.m
computeRMSE.m
GCMR.m
BreimanUnr.m
nuGCMR.m
fnuGKnock.m
functionknockoffCandes.m
predictionplotsGKnocktypes.m
predictionplotsunr.m
BostonHousingModels.mat
main_BostonHousing_Shuffled_20251024.m

Note:
knockoff experiments requires installation of MATLAB knockoffs.
Following the following link for installation: https://web.stanford.edu/group/candes/knockoffs/software/knockoffs/download-r.html


Main body:

Fig. 1: A visual illustration of the steps implied by Algorithm 1 to accompany Example 1.
Code: FigureGCMR.m

Fig. 2: Original (X) and permuted data (X′) for Example 1.
Code: FigureGCMR.m (storytelling_illust2.mlx)

Fig. 3: Discrete case.
Code: Figure4discrete.m

Fig. 4: Knockoffs transformation illustration.
Code: Figure5_storytelling_knockoff

Fig. 5: Hooker case.
Code: Figure7_HookerCase_BreiGCMRGKnock.m

Fig. 6: Variable Importance Evaluation on Generated Data. --- !!!
Code: 
- Moon Model missing
- Model 99 missing
- Model500_replicates_50000_App.m -- !! note: this code includes Wasserstein experiments.

Fig. 7: Boston Housing.
Code: main_BostonHousing_replicates_App_250812.m -- !! note: this code includes Wasserstein experiments./Figure saving codes commented out. 

Fig. 8: Empirical densities for Boston Housing.
Code: main_BostonHousing_Shuffled_20251024.m -- need to be cleaned.

Fig. 9: GPR-based Variance-Based c-SHAPs for the Boston Housing dataset.
Code: BostonHousing_Github.ipynb

Fig. 10: Breiman’s restricted permutation importance measures.
Code: MainEthnicity_NN.m