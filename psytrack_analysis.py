def psytrack_analysis(): 
       
    import psytrack as psy
    import pickle
     
    # choice of animals to load
  
    animals = ['c_129', 'c_136', 'c_137', 'opto_144', 'opto_145', 'opto_96', 'opto_97']
    
    
    psy_data = [];
    evd = [];
    
    # iterate through each animal and fit a PsyTrack model based on the 
    for a in animals:
        print(a)
        
        #model_evd, p = psytrack_model_choice(a)
        #psy_data.append(p);
        #evd.append(model_evd);
        m = load_and_clean_mat(a)
        [regressors, weights] = extract_regressors(m, r_choice = False)
        
        
        weight_choice = ['high', 'low', 'previous choice', 'previous correct',  'led', 'previous led', 'previous no go'];
        #weight_choice =  ['high', 'low', 'previous choice', 'previous correct', 'reaction time', 'led'];
        #weight_choice =  ['high', 'low', 'previous choice', 'previous correct', 'led'];
        regressors['inputs'] = { your_key: regressors['inputs'][your_key] for your_key in weight_choice };
        weight_choice.append('bias')
        weights = { your_key: weights[your_key] for your_key in weight_choice };
        k = len(weights.keys())
        
        hyper = {'sigInit': 2**4., 'sigma': [2**4.]*k, 'sigDay': 2**4.};
        optList = ['sigma', 'sigDay'];
        hyp, evd, wMode, hess_info = psy.hyperOpt(regressors, hyper, weights, optList, showOpt = 0); 
        p = {'hyp':hyp, 'evd':evd, 'wMode': wMode, 'animal': a};
        psy_data.append(p)
    
    save_data = {'psy_data': psy_data}
    f = open("lowhigh_dir_CAMKII_CONTROL_noRXN.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(save_data,f)

    # close file
    f.close()
          

def psytrack_model_choice(a):
    import psytrack as psy
    import itertools
    import numpy as np 
    
    model_evd = []
    m = load_and_clean_mat(a)
    [regressors, weights] = extract_regressors(m)
    t = regressors['inputs']['led'].shape[0];
    
    
    
    k = len(weights.keys())
    hyper = {'sigInit': 2**4., 'sigma': [2**4.]*k, 'sigDay': 2**4.};
    optList = ['sigma', 'sigDay'];
    hyp, full_model, wMode, hess_info = psy.hyperOpt(regressors, hyper, weights, optList, showOpt = 0); 
    mod_e = (-2 * full_model) + (len(optList) * np.log(t))
    model_evd.append(mod_e)
    p = {'hyp':hyp, 'evd':full_model, 'wMode': wMode, 'weights': weights}
    
    optList = ['sigma']
    hyp, evd, wMode, hess_info = psy.hyperOpt(regressors, hyper, weights, optList, showOpt = 0); 
    mod_e = (-2 * evd) + (len(optList) * np.log(t));
    model_evd.append(mod_e);
    if mod_e <= min(model_evd):
        p = {'hyp':hyp, 'evd':evd, 'wMode': wMode, 'weights':weights}
    
    weight_choice = itertools.combinations([*weights.keys()][2:6], 3);
    
    for c in weight_choice:
        wc = ['stimulus magnitude','stimulus direction']
        wc.extend(c);
        r = regressors;
        i = { your_key: r['inputs'][your_key] for your_key in wc };
        
        wc.append('bias')
        w = { your_key: weights[your_key] for your_key in wc };
        k = len(w.keys())
    
        optList = ['sigma', 'sigDay'];
        hyper = {'sigInit': 2**4., 'sigma': [2**4.]*k, 'sigDay': 2**4.};
        hyp, evd, wMode, hess_info = psy.hyperOpt(r, hyper, w, optList, showOpt = 0); 
        
        mod_e = (-2 * evd) + (len(optList) * np.log(t));
        model_evd.append(mod_e);
        if mod_e <= min(model_evd):
            p = {'hyp':hyp, 'evd':evd, 'wMode': wMode, 'weights':w}
            
            
        optList = ['sigma']
        hyper = {'sigInit': 2**4., 'sigma': [2**4.]*k, 'sigDay': None};
        hyp, evd, wMode, hess_info = psy.hyperOpt(r, hyper, w, optList, showOpt = 0); 
        mod_e = (-2 * evd) + (len(optList) * np.log(t));
        model_evd.append(mod_e);
        if mod_e <= min(model_evd):
            p = {'hyp':hyp, 'evd':evd, 'wMode': wMode, 'weights':w}
        
    return model_evd, p
    



def load_and_clean_mat(a):

    import numpy as np
    from scipy.io import loadmat
    import os
    
    folderpath = r'W:\Data\2AFC_Behavior';
    file_data = 'analyze_animal_'+ a +'.mat'
    path = os.path.join(folderpath,a, file_data);
    mouse_mat = loadmat(path, struct_as_record = False, squeeze_me = True, mat_dtype = True);
    m = mouse_mat['animal']
        
    file_training = 'analyze_training_'+a+'.mat'
    path = os.path.join(folderpath,a, file_training);
    mouse_train = loadmat(path,struct_as_record = False, squeeze_me = True, mat_dtype = True);
    t = mouse_train['training']
    
    
    m.LED = np.reshape(m.LED[round(t.trials_opto) : round(t.trials_expert)], ((round(t.trials_expert) - round(t.trials_opto)),1))
    m.lick = m.lick[:,round(t.trials_opto) : round(t.trials_expert)] 
    m.sessionNum = np.reshape(m.sessionNum[round(t.trials_opto) : round(t.trials_expert)], ((round(t.trials_expert) - round(t.trials_opto)),1))
    m.stimulus = np.reshape(m.stimulus[round(t.trials_opto) : round(t.trials_expert)], ((round(t.trials_expert) - round(t.trials_opto)),1))
    m.target = np.reshape(m.target[round(t.trials_opto) : round(t.trials_expert)], ((round(t.trials_expert) - round(t.trials_opto)),1))
    m.rxnTime = np.reshape(m.rxnTime[round(t.trials_opto) : round(t.trials_expert)], ((round(t.trials_expert) - round(t.trials_opto)),1))
    
    return m

def extract_regressors(m, r_choice):
    
    import numpy as np
    from scipy.stats import zscore
    
    lick = np.array(m.lick)
    rightward = lick[0][:] + lick[3][:] - lick[1][:] - lick[2][:]
    previous_choice = np.reshape(np.append(0, rightward[0:-1]), (len(rightward),1))
    answer = np.array(m.target)
    led = m.LED
    previous_led = np.reshape(np.append(0, led[0:-1]), (len(led),1))
    correct = lick[0][:] + lick[1][:]
    previous_nogo = np.reshape(np.append(0, lick[-1][0:-1]), (len(correct),1))
    
   
    previous_correct = np.reshape(np.append(0, correct[0:-1]), (len(correct),1))
    new_session = np.array(np.append(1, np.diff(m.sessionNum.T)), dtype = bool)
    previous_choice[new_session] = 0
    previous_led[new_session] = 0
    previous_correct[new_session] = 0
    previous_nogo[new_session] = 0
    
    go_trials = np.array(np.sum(lick[0:4],0), dtype = bool)

    #reaction = zscore(m.rxnTime[go_trials])
    # instead of zscoring rxnTime, normalized between 0  and 1
    reaction = (m.rxnTime[go_trials] - np.min(m.rxnTime[go_trials])) / (np.max(m.rxnTime[go_trials])- np.min(m.rxnTime[go_trials]))
    [sess, day_length] = np.unique(m.sessionNum[go_trials], return_counts=True)
    stimulus = np.log2(m.stimulus/8)/2
    stimulus[np.isneginf(stimulus)] = 0;
    stimulus[np.isposinf(stimulus)] = 0;
    
    if r_choice:
        stimulus_direction = np.array(stimulus > 0, dtype = 'int16') - np.array(stimulus < 0, dtype = 'int16')
        stimulus_magnitude = np.abs(stimulus)
        inputs = {'stimulus direction': stimulus_direction[go_trials], 'stimulus magnitude': stimulus_magnitude[go_trials], 'previous choice': previous_choice[go_trials], 'previous no go': previous_nogo[go_trials], 'previous led': previous_led[go_trials], 'previous correct': previous_correct[go_trials], 'reaction time': reaction, 'led': led[go_trials]};
    else:
        high = stimulus.copy();
        low = stimulus.copy();
        high[high < 0] = 0;
        low[low > 0] = 0;
        low = np.abs(low);
        inputs = {'high': high[go_trials], 'low': low[go_trials], 'previous choice': previous_choice[go_trials], 'previous correct': previous_correct[go_trials],'previous no go': previous_nogo[go_trials], 'previous led': previous_led[go_trials], 'reaction time': reaction, 'led': led[go_trials]};
    #inputs = {'stimulus': stimulus[go_trials], 'stimulus direction': stimulus_direction[go_trials], 'stimulus magnitude': stimulus_magnitude[go_trials], 'previous choice': previous_choice[go_trials], 'previous correct': previous_correct[go_trials], 'reaction time': reaction, 'led': led[go_trials]};
    
    

    weights = {k: 1 for k in inputs}
    weights['bias'] = 1
    
    y = rightward[go_trials];
    y[y == 1] = 2;
    y[y == -1] = 1;
    
    regressors = {'inputs': inputs, 'answer': answer, 'correct': correct, 'y': y, 'dayLength': day_length}

    
    return regressors, weights


if __name__ == "__main__":
    psytrack_analysis()
    