#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Aug 30 10:06:36 2018

@author: David Meier

Implementation of the Fake News model by DC Brody and DM Meier

("How to model fake news")

Notes:
    
CPU parallelization via joblib.
Integration routines done using Monte Carlo.
Main program begins ~line 650

For faster, but more inaccurate, computation reduce MC_integration_outer_steps, 
MC_integration_inner_steps, as well as Ngrid.

In main program choose between mode 'cloud' (main number crunching, useful if cloud server is used);
'laptop' (main plotting, useful if results have been downloaded from cloud server);
'single location' (both number crunching and plotting carried out on same machine).

"""


import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from joblib import Parallel, delayed

def ind(a, b):
    'indicator function'
    b = b*np.ones(a.shape) #do this explicitly; would work automatically, too'
    return 1*(a >= b)

def funex(x, C, m):
    'fake news function; magnitude C, damping constant m, x argument array'
    boolean = (x >=0)
    x = x*boolean
    return C*x*np.exp(-m*x)

def funex_der(x, C, m):
    """
    derivative of fake news function; x argument array
    We set to zero when x < 0; this is the case not of interest
    """
    boolean = (x >=0)
    x = x*boolean
    return C*np.exp(-m*x) - C*m*x*np.exp(-m*x)
    

def dens_tau(mu, x): #allow x to be stacked list for noises > 1
    """
    Density of tau -interarrival times- could be changed for different interarrival statistics.
    """
    return mu*np.exp(-mu*x)


def info(s, X, t_array, BM, tau, C, m):
    '''info process; t_array the time array, BM the BM array, tau fake news time,
       C magnitude, m damping constant
       X a scalar
       In multinoise case tau is a list of fake news waiting times. 
       '''
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])

    info = s*X*t_array + BM + fake_news
    return info

def exp_expression(xi_n, t_n, tau, s, X, C, m):
    """
    computes the exp expression inside the posterior probability; X can be array
    Returns an array
    
    We subtract X_max in the argument of exp (knowing that this will cancel overall
    in the expression for posterior probabilities) in order to avoid numerical overflows.
    
    Allow tau to be stacked list now. 
    """
    X_max = np.max(X)
    #remove fake_news
    
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_times_k = fake_times[:, k].reshape(tau.shape[0], 1)
        fake_news = fake_news + funex(t_n - fake_times_k, C, m) * ind(t_n, fake_times_k)
        #fake_news dimension is (tau.shape[0], 1)
    arg = xi_n*s*(X - X_max) - 0.5*s**2*X**2*t_n - s*X*fake_news
    return np.exp(arg)

def post_prob(prior, xi_n, t_n, tau, s, X, C, m):
    """
    X an array of values, prior an array of corresponding 
    returns: An array of posterior probabilities
    """
    exp_expr = exp_expression(xi_n, t_n, tau, s, X, C, m)
    denominator = np.sum(prior * exp_expr, axis = 1, keepdims = True) #shape (tau.shape[0], 1)
    #denominator = (prior @ (exp_expr).T)[0, 0]
    numerator = prior * exp_expr #shape (tau.shape[0], prior.shape[1])
    posterior = numerator / denominator
    return posterior #shape (tau.shape[0], prior.shape[1])

def F(prior, xi_n, t_n, tau, s, X, C, m):
    """
    The F function appearing in the integral
    xi_n and t_n scalars
    X an array
    """
    post = post_prob(prior, xi_n, t_n, tau, s, X, C, m)
    return np.sum(X * post, axis = 1, keepdims = True) #shape (tau.shape[0], 1)

def BM(t_array):
    """
    Create BM for the time array given
    """
    n = t_array.shape[1]
    step = t_array[0, 1] - t_array[0, 0]
    increments = np.sqrt(step) * np.random.randn(1, n-1)
    BM = np.cumsum(increments).reshape((1, increments.shape[1]))
    BM = np.insert(BM, 0, 0, axis = 1)
    return BM

def gen_t_array(T, N):
    """
    Create the time array for final time T and N steps
    """
    t_array = np.linspace(0, T, N+1).reshape((1, N+1))
    return t_array

def info_incr(info):
    """
    returns: Array of increments of info process
    """
    n = info.shape[1]
    info_shift = info[0, 1:n]
    info = info[0, 0:n-1]
    increments = info_shift - info
    return increments

def exp_eq33(y, info, t_array, s, X, C, m, index):
    """
    Computing right hand side of equation 33, June 2018 note
    index is the time index relative to which we condition
    X a scalar; y list of increments (although we now allow stacked lists)
    """
    fake_times = np.cumsum(y, axis = 1)
    fake_news_der = 0
    for k in range(y.shape[1]):
        fake_times_k = fake_times[:, k].reshape(y.shape[0], 1)
        fake_news_der = fake_news_der + funex_der(t_array - fake_times_k, C, m) * ind(t_array, fake_times_k)
    
    step = t_array[0, 1] - t_array[0, 0]
    a = -s*X*step + step*fake_news_der
    a = a[:, 0:index-1].reshape((y.shape[0], index-1))
    increments = info_incr(info).reshape(1, info.shape[1]-1)
    
    b = increments[0, 0:index-1] - a
    return np.exp((-1/(2*step) * np.sum(b*b, axis = 1, keepdims = True))) #returning a (y.shape[0], 1) array
        
def tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Posterior density for tau, but not yet normalized
    index is the time index relative to which we condition
    y stacked list
    """
    sum_exp = 0
    K = X_array.shape[1]
    
    for k in range(K):
        X = X_array[0,k]
        temp = exp_eq33(y, info, t_array, s, X, C, m, index)
        sum_exp = sum_exp + prior[0, k] * temp
    
    prior_densities = dens_tau(mu, y) #gives a stacked array which we need to take product of horizontally
    return np.prod(prior_densities, axis = 1, keepdims = True) * sum_exp #array of shape (y.shape[0], 1)

def integrand(y, info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Integrand in the final integral; this is a function of y (stacked list)
    index is the time index relative to which we condition (counting from 0!)
    """
    taup = tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index)
    
    xi_n = info[0,index]
    t_n = t_array[0,index]
    F_fun = F(prior, xi_n, t_n, y, s, X_array, C, m) #shape (y.shape[0], 1)
    
    integrand = taup * F_fun
    return integrand, taup #shape (y.shape[0], 1) for integrand; also return taup in order to get normalization. 

def cond_exp_own(info, t_array, s, X_array, C, m, prior, mu, index, noises, MC_measure, int_samples_in = 0):
    '''
    MC integration to compute conditional expectation
    '''
    integration_temps = []
    normalization_temps = []
    
    for k in range(MC_integration_outer_steps):
        summation = 0
        sum_tau_norm = 0
        if int_samples_in == 0: #then generate own integration samples
            int_samples = integration_upper_bound * np.random.rand(MC_integration_inner_steps, noises)
        else: #int_samples have been passed in as a list of arrays, list index the outer counter
            int_samples = int_samples_in[k]
        values, taup_vals = integrand(int_samples, info, t_array, s, X_array, C, m, prior, mu, index)
        summation = 1/MC_integration_inner_steps * np.sum(values, axis = 0)
        sum_tau_norm = 1/MC_integration_inner_steps * np.sum(taup_vals, axis = 0)
        integration_temps.append(summation)
        normalization_temps.append(sum_tau_norm)
                
    integral = 1/MC_integration_outer_steps * np.sum(integration_temps)
    factor = 1/MC_integration_outer_steps * np.sum(normalization_temps)
    return MC_measure * integral/factor, 0, 0 #not returning any errors for now; the measure factor in fact cancels with the measure factor in 'factor'

def plot_info(info_path, t_array, C, m, tau, candi, process, run = 1):
    """
    Plot information processes (two candidates)
    """
    time_axis = np.linspace(0, 1, N+1)
    plt.plot(time_axis, info_path[0], label = 'With fake news')
    #clean from fake news
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
    
    info_path_cleaned = info_path - fake_news
    plt.plot(time_axis, info_path_cleaned[0], label = 'Without fake news')
    if candi == 0:
        candiA = 'A'
    else:
        candiA = 'B'
    plt.title('Information process: candidate '+candiA+ ', process {}'.format(process+1))
    plt.xlabel('Time')
    plt.ylabel('Information process')
    plt.legend()
    if location == 'laptop' or location == 'single location':
        plt.savefig('Info_cand{}proc{}run{}.eps'.format(candi,process,run), format='eps', dpi=1000)

    plt.show()
    return 0

def get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau, noises, MC_measure, int_samples = 0): #can pass in int_samples in case want to fix MC integration variables
    """
    Compute best estimate (Category II -- or model2 -- voter)
    """
    best_estimate, relerror, parterror = cond_exp_own(info_path, t_array, s, X_array, C, m, prior, mu, index, noises, MC_measure, int_samples)
    return best_estimate , relerror, parterror

def true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau):
    '''
    Conditional expectation if fake news can be removed (same formula can be used for Category I -ignorant- voters)
    '''
    #clean from fake news
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
    
    info_path = info_path - fake_news
    info_current = info_path[0, index]
    time_current = t_array[0,index]
    
    list_exp = []
    K = X_array.shape[1]
    
    for k in range(K):
        X = X_array[0, k]
        temp = np.exp(info_current*s*X - 0.5 * s**2 * X**2 * time_current)
        list_exp.append(temp)
    
    exp_array = np.array(list_exp).reshape((1, K))
    denominator = prior @ exp_array.T
    numerator = X_array @ (prior * list_exp).T
    best_estimate = numerator / denominator
    
    return best_estimate[0, 0]
   
def gen_info(s, X_array, which_X, t_array, tau, C, m, run):
    """
    Generate information processes
    """
    info_processes = []
    for_file_plot = []
    for kx in range(len(tau)): #number of candidates
        
        X_array_current = X_array[kx]
        which_X_current = which_X[kx]
        tau_current = tau[kx]
        s_current = s[kx]
        C_current = C[kx]
        cand_info = []
        for ix in range(len(tau_current)): #number of processes
            X_array_proc = X_array_current[ix]
            which_X_proc = which_X_current[ix]
            which_tau_proc = tau_current[ix]
            s_proc = s_current[ix]
            C_proc = C_current[ix]
            
            BM_path = BM(t_array)
            info_path = info(s_proc, X_array_proc[0, which_X_proc], t_array, BM_path, which_tau_proc, C_proc, m)
            plot_info(info_path, t_array, C_proc, m, which_tau_proc, kx, ix, run)
            details = (info_path, t_array, C_proc, m, which_tau_proc, kx, ix, run)
            for_file_plot.append(details)

            cand_info.append(info_path)
        info_processes.append(cand_info)
    return info_processes, for_file_plot #list of lists [ [candA infoproc1, candA infoproc2, ...], [candB infoproc1, ...], ...]
    
def all_predictions(A):
    """
    Predictions are being generated for all three categories of voters.
    The main work is done for the Category II voters (here called model2). 
    """
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, index, noises, MC_measure) = A
    predictions = []
    for kx in range(len(tau)): #number of candidates
        X_array_current = X_array[kx]
        which_X_current = which_X[kx]
        tau_current = tau[kx]
        s_current = s[kx]
        info_current = info_processes[kx]
        prior_current = prior[kx]
        noises_current = noises[kx]
        MC_measure_current = MC_measure[kx]
        C_current = C[kx]
        pred_current = []
        for ix in range(len(tau_current)): #number of processes
            X_array_proc = X_array_current[ix]
            which_X_proc = which_X_current[ix]
            tau_proc = tau_current[ix]
            s_proc = s_current[ix]
            info_proc = info_current[ix]
            prior_proc = prior_current[ix]
            noises_proc = noises_current[ix]
            MC_proc = MC_measure_current[ix]
            C_proc = C_current[ix]
        
            model2_current, _, _ = get_best_estimate(info_proc, t_array, s_proc, C_proc, m, mu, prior_proc, X_array_proc, T, N, index, which_X_proc, tau_proc, noises_proc, MC_proc)
            ignorant_current = true_model_direct(info_proc, t_array, s_proc, 0, m, mu, prior_proc, X_array_proc, T, N, index, which_X_proc, tau_proc)
            true_current = true_model_direct(info_proc, t_array, s_proc, C_proc, m, mu, prior_proc, X_array_proc, T, N, index, which_X_proc, tau_proc)
            
            tup = (model2_current, ignorant_current, true_current)
            pred_current.append(tup) #pred_current list of predictions for candidate
        predictions.append(pred_current) #list of lists of predictions for all candidates
        
    return predictions

def Par_prediction_all_curve(B):
    """
    Setting things up to use Parallel from joblib (CPU parallelization).
    Predictions are computed for Ngrid points in total. 
    Ncore of which are done in parallel, if Ncore number of available CPUs.
    """
    #building Ngrid grid
    list_i = []
    conditioning_grid = np.linspace(1/Ngrid, 1, Ngrid)
    for k in range(Ngrid):
        indexi = int(conditioning_grid[k]*N)
        list_i.append(indexi)
        
    list_t = [B for i in list_i]
    
    results = Parallel(n_jobs=-1)(delayed(Par_allpred_helper)(i,t) for i,t in zip(list_i,list_t))  
    return results

def Par_allpred_helper(i, t):
    """
    Consistent with Parellel from joblib (CPU parallelization).
    Needs to have correct return behaviour.
    """
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, z) = t
    index = i
    A = (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, index, noises, MC_measure)
    res = all_predictions(A)
    return res

def get_all_pred(B_with_info_plot):
    """
    Gateway to the parallel computations
    """
    
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, info_plotting, Ncurve_index) = B_with_info_plot
    B = (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, Ncurve_index)
    
    results = Par_prediction_all_curve(B)
    
    output = open('result_{}.pkl'.format(Ncurve_index), 'wb')
    pickle.dump((results, B_with_info_plot), output) 
    output.close()

def plotting_pred_all():
    """
    Plotting functions.
    The ones that are not needed can be commented out for speed.
    """
    average_m2 = 0
    average_ig = 0
    average_true = 0
    
    #generating population
    Npop_slice = 100000
    n_of_slices = int(Npop/Npop_slice)
    
    W_dim1 = {}
    W_dim2 = {}
    W_dim3 = {}
    weight = 0.55
    
    #Dimension 1
    for i in range(n_of_slices):
        W_array = 0.4*np.random.randn(1, int(Npop_slice * weight)) + 1
        #45 percent here
        weight_dash = 1-weight
        W_array_other = 0.4*np.random.randn(1, int(Npop_slice * weight_dash)) - 1
        W_dim1["Slice_{}".format(i)] = np.concatenate((W_array, W_array_other), axis = 1)
    
    #Dimension 2
    for i in range(n_of_slices):
        #for each slice we must generate Npop_slice voters
        counter = 0
        weight_components = []
        while counter < int(weight * Npop_slice):
            trial = 0.4*np.random.randn() + 1
            if trial > 0:
                weight_components.append(trial)
                counter = counter + 1
        W_array = np.array(weight_components).reshape(1, int(Npop_slice*weight))
        #do again for other part of the population
        counter = 0
        weight_components = []
        while counter < int((1-weight) * Npop_slice):
            trial = 0.4*np.random.randn() + 1
            if trial > 0:
                weight_components.append(trial)
                counter = counter + 1
        W_array_other = np.array(weight_components).reshape(1,int( Npop_slice*(1-weight)))
        W_dim2["Slice_{}".format(i)] = np.concatenate((W_array, W_array_other), axis = 1)
        
    #Dimension 3
    for i in range(n_of_slices):
        #for each slice we must generate Npop_slice voters
        counter = 0
        weight_components = []
        while counter < int(weight * Npop_slice):
            trial = 0.4*np.random.randn() + 1
            if trial > 0 and trial < 1:
                #want between 0 and 1
                weight_components.append(trial)
                counter = counter + 1
        W_array = np.array(weight_components).reshape(1, int(Npop_slice*weight))
        #do again for other part of the population
        counter = 0
        weight_components = []
        while counter < int((1-weight)*Npop_slice):
            trial = 0.4*np.random.randn()
            if trial > 0 and trial < 1:
                weight_components.append(trial)
                counter = counter + 1
        W_array_other = np.array(weight_components).reshape(1,int( Npop_slice*(1-weight)))
        W_dim3["Slice_{}".format(i)] = np.concatenate((W_array, W_array_other), axis = 1)
        
    #keep same over runs to avoid spurious fluctuations
    W_array_dict = {}
    #bring it all together
    for i in range(n_of_slices):
        a = W_dim1["Slice_{}".format(i)]
        b = W_dim2["Slice_{}".format(i)]
        c = W_dim3["Slice_{}".format(i)]
        
        plt.hist(a[0, :])
        plt.show()
        
        plt.hist(b[0, :])
        plt.show()
        
        plt.hist(c[0, :])
        plt.show()
        
        d = np.concatenate((a, b), axis = 0)
        d = np.concatenate((d, c), axis = 0)
        
        W_array_dict["Slice_{}".format(i)] = d
        
    for z in range(Ncurves):
        time_axis = np.linspace(0, 1, Ngrid)
        print(z)
    
        pkl_file = open('Data_for_paper/result_{}.pkl'.format(z), 'rb')
        (results, B) = pickle.load(pkl_file)
        pkl_file.close()
        
        (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, for_info_plotting, Ncurve_index) = B
        #plotting information processes
        
        for rx in range(len(for_info_plotting)): 
            (info_path, t_array, C_proc, m, which_tau_proc, kx, ix, run) = for_info_plotting[rx]
            #plot_info(info_path, t_array, C_proc, m, which_tau_proc, kx, ix, run)
        
        #reset variables to what came in through B again
        (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, for_info_plotting, Ncurve_index) = B
        
        cand = {}
        differences = {} #differences between the curves (assuming just two candidates)
        
        #initializing dict
        for kx in range(len(tau)): #number of candidates
            tau_current = tau[kx]
            for ix in range(len(tau_current)): #number of processes
                cand["cand{}proc{}m2".format(kx, ix)] = []
                cand["cand{}proc{}ig".format(kx, ix)] = []
                cand["cand{}proc{}true".format(kx, ix)] = []
                
        #filling dict
        for i in range(len(results)): #Ngrid position i
            pred_curr = results[i]
            for kx in range(len(tau)): #number of candidates
                tau_current = tau[kx]
                pred_can = pred_curr[kx]
                for ix in range(len(tau_current)): #number of processes
                    pred_proc = pred_can[ix]
                    (model2_current, ignorant_current, true_current) = pred_proc
                    cand["cand{}proc{}m2".format(kx, ix)].append(model2_current)
                    cand["cand{}proc{}ig".format(kx, ix)].append(ignorant_current)
                    cand["cand{}proc{}true".format(kx, ix)].append(true_current)
        
        #plotting prediction curves
        for kx in range(len(tau)): #number of candidates
                tau_current = tau[kx]
                pred_can = pred_curr[kx]
                if kx == 0:
                    canda = 'A'
                else:
                    canda = 'B'
                for ix in range(len(tau_current)): #number of processes
                    plt.plot(time_axis, cand["cand{}proc{}ig".format(kx, ix)], label = "Category I")
                    plt.plot(time_axis, cand["cand{}proc{}m2".format(kx, ix)], label = "Category II")
                    plt.plot(time_axis, cand["cand{}proc{}true".format(kx, ix)], label = "Category III")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Estimates")
                    plt.title("Estimates: candidate "+canda+", process {}".format(ix+1))
                    if location == 'laptop' or location == 'single location':
                        plt.savefig('Estimate_cand{}proc{}run{}.eps'.format(kx, ix, z), format='eps', dpi=1000)
                    plt.show()
                    
        if len(tau) == 2: #number of candidates is 2
            for ix in range(len(tau_current)): #number of processes
                #differences a dictionary of the differences between processes
                differences["proc{}m2".format(ix)] = np.array(cand["cand0proc{}m2".format(ix)]) - np.array(cand["cand1proc{}m2".format(ix)])
                differences["proc{}ig".format(ix)] = np.array(cand["cand0proc{}ig".format(ix)]) - np.array(cand["cand1proc{}ig".format(ix)])
                differences["proc{}true".format(ix)] = np.array(cand["cand0proc{}true".format(ix)]) - np.array(cand["cand1proc{}true".format(ix)])
                
            for ix in range(len(tau_current)): #number of processes
                plt.plot(differences["proc{}m2".format(ix)], label = "Model2")
                plt.plot(differences["proc{}ig".format(ix)], label = "Ignorant")
                plt.plot(differences["proc{}true".format(ix)], label = "True")
                plt.legend()
                plt.title("Differences of processes {}".format(ix))
                plt.show()
                
            diff_m2_array = np.zeros((Nproc, Ngrid))
            diff_ig_array = np.zeros((Nproc, Ngrid))
            diff_true_array = np.zeros((Nproc, Ngrid))
            #generating multi-dim arrays of all process differences for two candidates
            for rx in range(Nproc):
                diff_m2_array[rx, :] = differences["proc{}m2".format(rx)]
                diff_ig_array[rx, :] = differences["proc{}ig".format(rx)]
                diff_true_array[rx, :] = differences["proc{}true".format(rx)]
                
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if len(tau) ==2 and len(tau_current) == 3: #have three difference processes; can visualize
                ax.plot(diff_ig_array[0, :], diff_ig_array[1, :], diff_ig_array[2, :], label = 'Category I')
                ax.plot(diff_m2_array[0, :], diff_m2_array[1, :], diff_m2_array[2, :], label = 'Category II')
                ax.plot(diff_true_array[0, :], diff_true_array[1, :], diff_true_array[2, :], label = 'Category III')
                ax.xaxis._axinfo['label']['space_factor'] = 15
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.yaxis.set_tick_params(labelrotation = -15)
                ax.zaxis.set_tick_params(labelrotation = 0)
                ax.set_xlabel("Process 1")
                ax.set_ylabel("Process 2")
                ax.set_zlabel("Process 3")
                plt.legend()
                
                if location == 'laptop' or location == 'single location':
                    plt.savefig('Diffs_run{}.eps'.format(z), format='eps', dpi=1000)
                
                plt.show()
            
            votes_m2 = 0
            votes_ig = 0
            votes_true = 0
            for sx in range(n_of_slices): 
                W_array = W_array_dict["Slice_{}".format(sx)]
                
                votes_m2_array = W_array.T @ diff_m2_array
                votes_ig_array = W_array.T @ diff_ig_array
                votes_true_array = W_array.T @ diff_true_array
                
                votes_m2_array = votes_m2_array > 0
                votes_ig_array = votes_ig_array > 0
                votes_true_array = votes_true_array > 0
                
                votes_m2 = votes_m2 + np.sum(votes_m2_array, axis = 0, keepdims = True)/Npop
                votes_ig = votes_ig + np.sum(votes_ig_array, axis = 0 , keepdims = True)/Npop
                votes_true = votes_true + np.sum(votes_true_array, axis = 0 , keepdims = True)/Npop
                    
            plt.plot(time_axis, votes_ig[0, :], label = "Category I")
            plt.plot(time_axis, votes_m2[0, :], label = "Category II")
            plt.plot(time_axis, votes_true[0,:], label = "Category III")
           
            plt.xlabel("Time")
            plt.ylabel("Vote for candidate A")
            plt.legend()
            plt.title("Voting proportions")
            if location == 'laptop' or location == 'single location':
                plt.savefig('Vote_prop_run{}.eps'.format(z), format='eps', dpi=1000)
            plt.show()
            
            #store in average
            average_m2 = average_m2 + votes_m2
            average_ig = average_ig + votes_ig
            average_true = average_true + votes_true
            
        if len(tau_current) == 2 and len(tau) == 2: #number of processes is 2 -->> can visualize
            plt.plot(differences["proc0ig"], differences["proc1ig"], label = "Category I")
            plt.plot(differences["proc0m2"], differences["proc1m2"], label = "Category II")
            plt.plot(differences["proc0true"], differences["proc1true"], label = "Category III")
            plt.legend()
            plt.title("X_A - X_B")
            plt.show()
            
    print(average_true/Ncurves)
    print(average_m2/Ncurves)
    print(average_ig/Ncurves)
    plt.plot(time_axis, average_ig[0, :]/Ncurves, label = 'Category I')
    plt.plot(time_axis, average_m2[0, :]/Ncurves, label = 'Category II')
    plt.plot(time_axis, average_true[0, :]/Ncurves, label = 'Category III')
    plt.xlabel("Time")
    plt.ylabel("Vote for candidate A")
    plt.legend()
    plt.title('Average of voting proportions over {} runs'.format(Ncurves))
    if location == 'laptop' or location == 'single location':
        plt.savefig('Average.eps', format='eps', dpi=1000)
    plt.show()
    


"""
Main program begins here. 

"""
    
"""
Global variables, random seed, etc
"""
#np.random.seed(0) #may enable for testing

location = 'cloud' #'laptop' or 'cloud' or 'single location'
#'single location' if one machine is used for computations and plotting
#'cloud' if the machine does the computation and then pickles the result
#'laptop' if the machine does the unpickling and plotting


integration_upper_bound = 1  #for own integration routines with noises = 1 as well as MC bounds for uniform distribution from which we draw integration variable samples
MC_integration_outer_steps = 200 #number of MC steps we do using a for loop
MC_integration_inner_steps = 200 #number of (length of tau)-dimensional integration variables we use in MC in vectorized computation (function values computed all at once)
#Total number of MC samples is the product MC_integration_outer_steps * MC_integration_inner_steps


Npop = 100000 #size of population modelled; should be multiple of 100'000
assert(Npop%100000) == 0 #100K is the size of a slice in plotting to keep it manageable

"""
Parameters and initializations
"""

s = [[.2, .2, .2], [.2, .2, .2]] #information flow rates [(candA factor1, candA factor2), (candB factor1, candB factor2), etc]
C = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]] #magnitude of fake news
m = 4 #decay rate of fake news
mu = 4 #rate parameter of the exponential distribution for waiting times

prior = [[np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])], [np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])]]#prior probabilities
X_array = [[np.array([[-1, 1]]), np.array([[-1, 1]]), np.array([[-1, 1]])] , [np.array([[-1, 1]]), np.array([[-1, 1]]), np.array([[-1, 1]])]] #the true underlying values we attempt to detect
T = 1 #final time
N = 512 #number of steps on the time axis when discretizing the information process


which_X =  [ [1, 1, 1], [0, 0, 0]] #tells us index at which value of X_array is the true one; that is, true value is X_array[0, which_X]
Ngrid = 16 #number of steps discretizing time for the prediction curves 
Nproc = len(which_X[0]) #number of processes per candidate; some of the plotting only works when this equals 2
Ncurves = 100 #number of curves we draw to get average

t_array = gen_t_array(T, N) #generate t_array
        
total_time = 0
start = time.time()

max_noises = 10 #the mathematics is designed so that we have to fix the maximal number of noises we are willing to consider for each process
                # the MC integration is then over a cube of tau values of this dimension
                # would like this number to be large enough so that probability of more fake news items is small

for z in range(Ncurves):
    if location == 'cloud' or location == 'single location':
        
        #build up random tau -- interarrival times
        dict_tau = {}
        for prx in range(2*Nproc): #need taus for the 6 processes
            waiting_times = []
            for noise_indx in range(max_noises):
                waiting_times.append(np.random.exponential(1/mu))
            dict_tau["pr{}".format(prx)] = np.array(waiting_times).reshape(1, len(waiting_times))
            
        tau = [[dict_tau["pr0"], dict_tau["pr1"], dict_tau["pr2"]], [dict_tau["pr3"],  dict_tau["pr4"], dict_tau["pr5"]     ]]
        
        #build up number of noises and MC_measures for each process
        #this code is a left-over from an initial version; could be simplified. We're just constructing a list of lists.
        noises = []
        MC_measure = []
        for kx in range(len(tau)): #number of candidates
            temp = tau[kx]
            templist = []
            templist_MC = []
            for ix in range(len(temp)): #number processes
                templist.append(max_noises) #keep max_noises constant!
                templist_MC.append(integration_upper_bound**max_noises)
            noises.append(templist) #number of fake news items
            MC_measure.append(templist_MC) #MC measures for each process of each candidate; volume used to normalize MC integral (normalization for multivariable probability density)
        
        #generate information processes
        info_processes, for_file_plot = gen_info(s, X_array, which_X, t_array, tau, C, m, z)  
    
        #summarize all parameters
        B = (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau, info_processes, noises, MC_measure, for_file_plot, z)
    
        get_all_pred(B) # get predictions
        
        print(z)
        endvar = time.time()
        total_time = total_time + endvar - start
        print("Done {} percent".format(z/Ncurves * 100))
        print("ETA is {} minutes".format((endvar - start)*(Ncurves - 1 - z)/60)) #depends on timing of each cycle; gives an approximately correct estimate
        start = time.time()
        
print(total_time)

if location == 'laptop' or location == 'single location':
    plotting_pred_all()

endvar = time.time()
print(endvar-start)


    
    
    



    
    
    
    
    
    
    
    





