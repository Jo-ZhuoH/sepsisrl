# sepsisrl
Reinforcement Learning for optimal sepsis treatment policies

[1] Deep Reinforcement Learning for Sepsis Treatment

[2] Continuous State-Space Models for Optimal Sepsis Treatment: a Deep Reinforcement Learning Approach

## /preprocessing/

### 1. process_interventions
`MKdataset07Feb17` --> `discretised_input_data`

- Define and add `iv_input`, `vaso_input`

### 2. preprocess_data
`discretised_input_data` --> `rl_train_set_unscaled`, `rl_train_set_scaled`, ... (train:val:test=7:1:2)

- values deemed excessively high/low are capped
- Add sparsely rewards
    - `died_in_hosp` 1: -100, 0: 100
    -  0:224552, 100:15583, -100:2315
- Relevant binary features and normally/log-normally features are standardised accordingly
- Training and test sets are split - 70% train (n=169495), 10% validation (n=24338), 20% test (n=48617). all n = 242450
- binary_fields = ['gender','mechvent','re_admission']
- norm_fields = ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1','Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium','Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2','Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index', 'PaO2_FiO2','cumulated_balance_tev', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
- log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total_tev','input_4hourly_tev','output_total','output_4hourly', 'bloc']

### 3. Cluster 
`rl_train_set_unscaled`, `state_features` --> `rl_train_data_discrete`

- Kmeans on train_set (1250 centroids), predict cluster centroids for val and test data.
- get the state columns, which is the label of centroids
- only keep the `bloc`, `icustayid`, `state`, `reward`, `max_dose_vaso`, `iv_tev_in`, and `mortality`. in the `rl_train_data_discrete` file.

### 4. Rewards/Rewards no terminal
`rl_train_set_scaled` --> `rl_train_data_final_cont`/ `rl_train_data_final_cont_noterm`

- **range(-100/+100) --> range(-15/+15)**
- **Rewards no terminal**: 
    - c0 = -0.25, c1 = -0.75, c2 = -0.5
    - if sofa_cur(post?) == sofa_prev(pre, kind of), reward += c0
    - reward += c1*(sofa_cur-sofa_prev) # TODO: similar to CRT, LVEF if increased > 5% add reward
    - reward += c2*np.tanh(lact_cur - lact_prev) # TODO: why use tanh here?
- **Reward with terminal**: add reward=0 to the previouse reward (-15/+15) (hist: Gaussian distribution)
    - c0=-0.1/4, c1=-0.5/4, c2=-2
    - if sofa_cur(post?) == sofa_prev(pre, kind of), reward += c0
    - reward += c1*(sofa_cur-sofa_prev) # TODO: similar to CRT, LVEF if increased > 5% add reward
    - reward += c2*np.tanh(lact_cur - lact_prev) # TODO: why use tanh here?
    

## /continuous/

### 1. autoencoder
`rl_train_data_final_cont` --> `rl_train_data_final_auto`
**Get a reduced dimensionality representation**

- input the features in `state_features` file into autoencoder.
- add actions, reward, and icuid into autoencoder dataframe later.
- save to `rl_train_data_final_auto` file.

### 2. q_network
`rl_train_data_final_cont_noterm` --> 

- Get state_features.txt
- REWARD_THRESHOLD = 20, reg_lambda = 5
- Important weights and params:
    - per_flag = True
    - beta_start = 0.9
    - prob = abs(reward)
    - temp = 1/prob
    - imp_weight = pow((1/len(df) * temp), beta_start)
    - hidden_1_size = 128
- Qnetwork
- Save results
    - 1000 steps: per_weights/imp_weight
    - 5000 steps: do_save_resutls(): 
        - `dqn_normal_actions_train.p` agent_actions_train
        - `dqn_normal_q_train.p` agent_q_train

### 3. q_network_autoencoder
train:`rl_train_data_final_auto`, train_orig:`rl_train_data_final_cont` -->

Uses these for later in policy evaluation and figures.
- `dqn_autoencode_actions_train.p`
- `dqn_autoencode_q_train.p`



    
            
