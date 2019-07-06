#######
#######
#######	IMPORT REQUIRED THINGS
#######
#######


from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import pandas as pd
import time
from scipy.optimize import minimize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle


#######
#######
#######	DEFINE ALL FUNCTIONS
#######
#######


#define functions

#create function that finds mean of a function
def mean_pop(a):
    mean = 0
    for i in range(len(a)):
        mean += a[i]*i

    mean = int(mean/sum(a))

    return mean

#create function that finds mean of a population
def weighted_total(a):
    weighted_total=0
    history_number=np.zeros((100,1)) #100 being the number of possible credit scores
    for i in range(100):
        number=sum(a==i) #number being the number of people having credit score i
        weighted_total+=number*i
        history_number[i]=number
    
    weighted_total/=sum(history_number)
    
    return weighted_total

#creates random distributions
def brownian_population(delta=2, dt=1, n=100, x=0):

    history=np.zeros(n)
    
    # Iterate to compute the steps of the Brownian motion.
    for i in range(n):
        x = x + norm.rvs(scale=delta**2*dt)
        history[i] = x
    
    history = history+np.abs(np.min(history))
    
    original_hist = np.copy(history)
    old_mean = mean_pop(original_hist)
    
    history = np.round(999*history/sum(history))
    
    mean = np.round(np.average(history))
    
    if sum(history) < 999:
        history[0] += 999-sum(history)
        
    elif sum(history) > 999:  
        history[np.argmax(history)] -= -999+sum(history)        
    

    return history, old_mean, original_hist

#create function that finds true positive rates as a function of the cutoff
def true_positive_rate(population, population_that_defaulted, cutoff):
    population_bank_thinks_defaulted = population < cutoff
    
    true_positives = np.logical_and(np.logical_not(population_that_defaulted), 
                                    np.logical_not(population_bank_thinks_defaulted))
    
    number_true_positives = sum(true_positives)
    
    false_negatives = np.logical_and(np.logical_not(population_that_defaulted), 
                                     population_bank_thinks_defaulted)
    
    number_false_negatives = sum(false_negatives) 
    
    true_positive_rate = number_true_positives/(number_true_positives+number_false_negatives)
    
    return true_positive_rate

def difference_bw_positive_rates(pop1, pop2, pop1_that_defaulted, 
                                 pop2_that_defaulted, cutoff1, cutoff2, strictness=0.01):
    true_positive_rate_group1 = true_positive_rate(pop1, pop1_that_defaulted, cutoff1)
    true_positive_rate_group2 = true_positive_rate(pop2, pop2_that_defaulted, cutoff2)
    # for the optimiser's constraints
    return strictness - np.abs(true_positive_rate_group1-true_positive_rate_group2)

def difference_bw_selection_rates(pop1, pop2, pop1_that_defaulted, 
                                  pop2_that_defaulted, cutoff1, cutoff2, strictness=0.01):
    
    _, selection_rate_group1, _ = bank_profit(pop1, pop1_that_defaulted, cutoff1)
    _, selection_rate_group2, _ = bank_profit(pop2, pop2_that_defaulted, cutoff2)
    
    return strictness - np.abs(selection_rate_group1-selection_rate_group2)

#create function that finds bank profit as a function of the cutoff
def bank_profit(population, population_that_defaulted, cutoff, 
                revenue_factor=1, cost_factor=3, credit_upside=1, credit_downside=2):
    population_bank_thinks_defaulted = population < cutoff
    
    true_positives = np.logical_and(np.logical_not(population_that_defaulted), 
                                    np.logical_not(population_bank_thinks_defaulted))
    
    paid_back = sum(true_positives)
    false_positives = np.logical_and(population_that_defaulted, 
                                     np.logical_not(population_bank_thinks_defaulted))
    defaulted = sum(false_positives)
    
    profit = paid_back*revenue_factor - defaulted*cost_factor
    selection_rate = sum(np.logical_not(population_bank_thinks_defaulted))/len(population)
    
    credit_change = (paid_back*credit_upside - defaulted*credit_downside)

    
    return profit, selection_rate, credit_change



def total_bank_profit(pop1, pop2, pop1_that_defaulted, pop2_that_defaulted, cutoff1, cutoff2, 
                      revenue_factor=1, cost_factor=2, credit_upside=1, credit_downside=2):
    
    profit_group1, _, _ = bank_profit(pop1, pop1_that_defaulted, cutoff1)
    profit_group2, _, _ = bank_profit(pop2, pop2_that_defaulted, cutoff2)
    
    return -(profit_group1 + profit_group2)

def create_blues_and_oranges():
    a, mean_a, _ = brownian_population()
    b, mean_b, _ = brownian_population()

    if mean_a >= mean_b:
        blues = np.copy(b)
        oranges = np.copy(a)
    elif mean_a < mean_b:
        blues = np.copy(a)
        oranges = np.copy(b)
    else:
        print("um")

    true_blues = np.zeros(1)
    true_oranges = np.zeros(1)
    
    for i in range(100):
        num_blues = int(blues[i])
        num_oranges = int(oranges[i])
        
        if num_blues >= 0 and num_oranges >= 0:

            elements_blues = np.multiply(np.ones(num_blues),i)
            elements_oranges = np.multiply(np.ones(num_oranges), i)

            true_blues = np.concatenate((true_blues, elements_blues), axis=0)
            true_oranges = np.concatenate((true_oranges, elements_oranges), axis=0)

    #determine who defaulted
    blues_who_defaulted = true_blues.T < np.random.uniform(0, 100, (1000,1)).ravel()
    oranges_who_defaulted = true_oranges.T < np.random.uniform(0, 100, (1000,1)).ravel()

    blues = true_blues.T
    oranges = true_oranges.T
    
    return blues, oranges, blues_who_defaulted, oranges_who_defaulted, mean_a, mean_b
    
#######
#######
#######	SET UP LOOP
#######
#######


how_many_iters_ugh = 500
how_much_history = 10

history_initials_blue = np.zeros((how_many_iters_ugh, 1000))
history_initials_orange = np.zeros((how_many_iters_ugh, 1000))

history_final_blue_fair = np.zeros((how_many_iters_ugh, how_much_history, 1000))
history_final_orange_fair = np.zeros((how_many_iters_ugh, how_much_history, 1000))
history_cutoff_fair = np.zeros((how_many_iters_ugh, how_much_history, 2))

history_final_blue_unfair = np.zeros((how_many_iters_ugh, how_much_history, 1000))
history_final_orange_unfair = np.zeros((how_many_iters_ugh, how_much_history, 1000))
history_cutoff_unfair = np.zeros((how_many_iters_ugh, how_much_history, 2))




#######
#######
#######	RUN LOOP
#######
#######


for j in range(how_many_iters_ugh):
    
    t = time.time()
    
    b_neutral, o_neutral, bwd_neutral, owd_neutral, mb, mo = create_blues_and_oranges()
    
    b_unfair = np.copy(b_neutral)
    o_unfair = np.copy(o_neutral)
    bwd_unfair = np.copy(bwd_neutral)
    owd_unfair = np.copy(owd_neutral)

    b_fair = np.copy(b_neutral)
    o_fair = np.copy(o_neutral)
    bwd_fair = np.copy(bwd_neutral)
    owd_fair = np.copy(owd_neutral)
    
    history_initials_blue[j] = b_neutral
    history_initials_orange[j] = o_neutral
    
    for i in range(how_much_history):

        #optimise cutoffs without fairness
        
        to_optimise = lambda x: total_bank_profit(b_unfair, o_unfair, bwd_unfair, 
                                                  owd_unfair, x[0], x[1])

        #initial guess
        cutoffs = np.zeros(2)
        cutoffs[0] = 50
        cutoffs[1] = 50


        #optimise without fairness
        a = (0, 100)
        bounds = (a, a)

        solution_unfair = minimize(to_optimise, cutoffs, 
                            bounds=bounds
                                   , options = {'eps': 3}
                                  )

        x_unfair = solution_unfair.x

        loans_b = b_unfair < x_unfair[0]
        loans_o = o_unfair < x_unfair[1]
        true_positives_b = np.logical_and(np.logical_not(bwd_unfair), np.logical_not(loans_b))
        false_positives_b = np.logical_and(np.logical_not(loans_b), bwd_unfair)

        true_positives_o = np.logical_and(np.logical_not(owd_unfair), np.logical_not(loans_o))
        false_positives_o = np.logical_and(np.logical_not(loans_o), owd_unfair)

        b_unfair[true_positives_b] +=5
        b_unfair[false_positives_b] -=10
        b_unfair[b_unfair<1] = 1
        b_unfair[b_unfair>99] = 99

        o_unfair[true_positives_o] +=5
        o_unfair[false_positives_o] -=10
        o_unfair[o_unfair<1] = 1
        o_unfair[o_unfair>99] = 99

        history_final_blue_unfair[j, i, :]=b_unfair
        history_final_orange_unfair[j, i, :]=o_unfair
        history_cutoff_unfair[i]=x_unfair

        bwd_unfair = b_unfair < np.random.randint(100, size=(1000,))
        owd_unfair = o_unfair < np.random.randint(100, size=(1000,))
    
    for i in range(how_much_history):
        
        #optimise cutoffs WITH fairness
        #what to optimise
        to_optimise = lambda x: total_bank_profit(b_fair, o_fair, bwd_fair, 
                                                  owd_fair, x[0], x[1])

        #what to constrain
        to_constrain = lambda x: difference_bw_positive_rates(b_fair, o_fair, bwd_fair, 
                                                              owd_fair, x[0], x[1])

        #initial guess
        cutoffs = np.zeros(2)
        cutoffs[0] = 50
        cutoffs[1] = 50

        a = (0, 100)
        bounds = (a, a)

        #name constraints
        equal_true_positives = {'type': 'ineq', 'fun': to_constrain}
        cons = ([equal_true_positives])



        solution_fair = minimize(to_optimise, cutoffs, 
                            bounds=bounds, constraints=cons
                                   , options = {'eps': 3, 'maxiter':10}
                                  )

        x_fair = solution_fair.x


        loans_b = b_fair < x_fair[0]
        true_positives_b = np.logical_and(np.logical_not(loans_b), np.logical_not(bwd_fair))
        false_positives_b = np.logical_and(np.logical_not(loans_b), bwd_fair)


        loans_o = o_fair < x_fair[1]
        true_positives_o = np.logical_and(np.logical_not(loans_o), np.logical_not(owd_fair))
        false_positives_o = np.logical_and(np.logical_not(loans_o), owd_fair)

        b_fair[true_positives_b] +=5
        b_fair[false_positives_b] -=10
        b_fair[b_fair<1] = 1
        b_fair[b_fair>99] = 99

        o_fair[true_positives_o] +=5
        o_fair[false_positives_o] -=10
        o_fair[o_fair<1] = 1
        o_fair[o_fair>99] = 99

        history_final_blue_fair[j,i,:]=b_fair
        history_final_orange_fair[j,i,:]=o_fair
        history_cutoff_fair[j,i,:]=x_fair
        

        bwd_fair = b_fair < np.random.randint(100, size=(1000,))
        owd_fair = o_fair < np.random.randint(100, size=(1000,))




pickled = [history_initials_blue, history_initials_orange, 
	history_final_blue_fair, history_final_orange_fair, history_cutoff_fair, 
	history_final_blue_unfair, history_final_orange_unfair, history_cutoff_unfair]

file_Name = "experiment_results2.pkl"
with open(file_Name, 'wb') as f:
	pickle.dump(pickled, f)


# with open(file_Name, 'rb') as f:
# 	unpickled = pickle.load(f)
# # history_initials_blue, history_initials_orange, 
# # 	history_final_blue_fair, history_final_orange_fair, history_cutoff_fair, 
# # 	history_final_blue_unfair, history_final_orange_unfair, history_cutoff_unfair = unpickled
# print(len(unpickled))
# test = unpickled[0]
# print(test[0, 0, :])