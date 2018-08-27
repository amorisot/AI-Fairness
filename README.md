# AI-Fairness

This repository contains all the experiments I performed over the course of my M.Sc. thesis. Most of them did not make it into the final cut. This readme explains what each repository contains, and results that, while not worthy of being included in the thesis, are still reasonably interesting. 

## datasets

I performed a few preliminary experiments on the german credit dataset, included in the datasets repository. These mostly highlight the extent to which sensitive features are entangled with non-sensitive features, i.e. the extent to which they are correlated. 

## notebooks

### notebooks/unused_results_and_miscellanea

Three things of note here:

* A logistic regression trained on the 20 dimensional feature vector of the german credit dataset can get somewhere around 80% accuracy. A logistic regression trained only on the sensitive feature and the results can get somewhere around 75% accuracy. This speaks to how correlated sensitive features are with other features. (`german_credit_playground.ipynb`)  

* I simulated a population of people with different means and plotted the effects of logistic regression, which made their means grow further apart while the regression's accuracy was increasing. This was before I had read Liu et al.'s paper, which provided a better framework for exploring the things I wanted to explore, and so didn't get incorportated into the final thesis. However, it did produce some fun plots. (`the_model_explores.ipynb`)  

* The most consequential thing that I didn't include were plots from my `is_enforcing_equalised_odds_good?.ipynb` notebook. Mostly, it involved relaxing the strictness of the fairness criteria, and examining the effects of this relaxation on the change in credit score of the population. I found that a more relaxed fairness criteria led to a higher score. I didn't include this in the final thesis mostly because it didn't fit neatly into the flow of my thesis, and the results are a little underpowered because they were obtained on only a few populations (as opposed to the hundreds it would have taken me to get real results), but it's still worth mentioning here.

### notebooks/10-step-experiments

The notebooks that I used to to figure out how to run a long simulation, the code of which was then ported into the scripts repository. These notebooks also contain some of the plots included in the thesis. Note that each of them takes about 8-10 hours to run.

### notebooks/single_step_experiments_w_different_params_changed

The notebooks that I used to run single step experiments. `brownian_motion.ipynb` is the one where I first simulated the population movement, and the others are for different parameters and fairness criteria etc. These can take some time to run, depending on how many individuals are simulated. 100 individuals takes around 5 minutes. 

## scripts

Simply the script version of the notebook in notebooks/10-step-experiments.

