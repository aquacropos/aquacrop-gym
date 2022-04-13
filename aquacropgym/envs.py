from aquacrop.classes import *
from aquacrop.core import *
 
import gym
from gym import spaces

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from aquacrop.lars import *
# import math


nebraska_maize_config = dict(
    name='nebraska_maize',
    gendf=None, # generated and processed weather dataframe
    year1=1, # lower bolund on train years
    year2=70, # upper bound on train years
    crop='Maize', # crop type (str or CropClass)
    planting_date='05/01',
    soil='SiltClayLoam', # soil type (str or SoilClass)
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    include_rain=True, # maximum number of days to simulate at start of season (ramdomly drawn)
    days_to_irr=7, # number of days (sim steps) to take between irrigation decisions
    max_irr=25, # maximum irrigation depth per event
    init_wc=InitWCClass(wc_type='Pct',value=[70]), # initial water content
    crop_price=180., # $/TONNE
    irrigation_cost = 1.,# $/HA-MM
    fixed_cost = 1728,
    best=np.ones(1000)*-1000, # current best profit for each year
    observation_set='default',
    normalize_obs=True,
    action_set='smt4',
    forecast_lead_time=7, # number of days perfect forecast if using observation set x
    evaluation_run=False,
    CO2conc=363.8,
    simcalyear=1995,

)

cordoba_cotton_config = dict(
    name='cordoba_cotton', 
    gendf=None, # generated and processed weather dataframe
    year1=1, # lower bolund on train years
    year2=70, # upper bound on train years
    crop='Cotton', # crop type (str or CropClass)
    planting_date='04/25',
    soil='SandyLoam', # soil type (str or SoilClass)
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    include_rain=True, # maximum number of days to simulate at start of season (ramdomly drawn)
    days_to_irr=7, # number of days (sim steps) to take between irrigation decisions
    max_irr=25, # maximum irrigation depth per event
    init_wc=InitWCClass(wc_type='Pct',value=[70]), # initial water content
    crop_price=330., # euro/TONNE
    irrigation_cost = 0.3,# euro/HA-MM
    fixed_cost = 0, # gross margin
    best=np.ones(1000)*-1000, # current best profit for each year
    observation_set='default',
    normalize_obs=True,
    action_set='smt4',
    forecast_lead_time=7, # number of days perfect forecast if using observation set x
    evaluation_run=False,
    CO2conc=363.8,
    simcalyear=1995,

)

california_tomato_config = dict(
    name='california_tomato', 
    gendf=None, # generated and processed weather dataframe
    year1=1, # lower bolund on train years
    year2=70, # upper bound on train years
    crop=CropClass('Tomato',PlantingDate='05/01',PlantPop=100_000), # crop type (str or CropClass)
    planting_date='05/01',
    soil='Loam', # soil type (str or SoilClass)
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    include_rain=True, # maximum number of days to simulate at start of season (ramdomly drawn)
    days_to_irr=7, # number of days (sim steps) to take between irrigation decisions
    max_irr=25, # maximum irrigation depth per event
    init_wc=InitWCClass(wc_type='Pct',value=[70]), # initial water content
    crop_price=81., # $/TONNE
    irrigation_cost = 0.5,# $/HA-MM
    fixed_cost = 0, # gross margin
    best=np.ones(1000)*-1000, # current best profit for each year
    observation_set='default',
    normalize_obs=True,
    action_set='smt4',
    forecast_lead_time=7, # number of days perfect forecast if using observation set x
    evaluation_run=False,
    CO2conc=363.8,
    simcalyear=1995,

)

class CropEnv(gym.Env):
 
    def __init__(self,config):

        
        super(CropEnv, self).__init__()
 
        self.gendf = config["gendf"]
        self.days_to_irr=config["days_to_irr"]
        self.eval= config['evaluation_run']
        self.year1=config["year1"]
        self.year2=config["year2"]
        if self.eval:
            self.chosen=self.year1
        else:
            self.chosen = np.random.choice([i for i in range(self.year1,self.year2+1)])
 
        self.dayshift = config["dayshift"]
        self.include_rain=config["include_rain"]
        self.max_irr=config["max_irr"]
        self.init_wc = config["init_wc"]
        self.CROP_PRICE=config["crop_price"]
        self.IRRIGATION_COST=config["irrigation_cost"] 
        self.FIXED_COST = config["fixed_cost"]
        self.planting_month = int(config['planting_date'].split('/')[0])
        self.planting_day = int(config['planting_date'].split('/')[1])

        crop = config['crop']        
        if isinstance(crop,str):
            self.crop = CropClass(crop,PlantingDate=config['planting_date'])
        else:
            assert isinstance(crop,CropClass), "crop needs to be 'str' or 'CropClass'"
            self.crop=crop

        soil = config['soil']
        if isinstance(soil,str):
            self.soil = SoilClass(soil)
        else:
            assert isinstance(soil,SoilClass), "soil needs to be 'str' or 'SoilClass'"
            self.soil=soil
     
        self.name=config["name"]

        self.best=config["best"]*1
        self.total_best=config["best"]*1
        self.tsteps=0

        self.simcalyear=config["simcalyear"]
        self.CO2conc=config["CO2conc"]
        
    
        self.observation_set=config["observation_set"]
        self.normalize_obs = config["normalize_obs"]
        self.action_set=config["action_set"]
        self.forecast_lead_time=config["forecast_lead_time"]

        if self.name=='cordoba_cotton':
            self.mean=np.array([1.5416185e+01, 7.2023120e+00, 4.8906943e-01, 8.8000000e+01,
                                3.0104053e+02, 5.2842396e-01, 4.6625201e+02, 7.3104918e-01,
                                6.6835184e+00, 5.9939682e+01, 6.0787720e+02,0,0,0,0], dtype=np.float32)
        
            self.std= np.array([8.9190435e+00, 1.6620870e+00, 2.2383636e-01, 4.9937759e+01,
                                2.2935059e+02, 3.7969297e-01, 4.3991928e+02, 1.8572671e+00,
                                1.3334695e+00, 5.1132626e+01, 3.5932901e+02, 1,1,1,1], dtype=np.float32)

        elif self.name =='nebraska_maize':
            if self.observation_set=='default':
                self.mean=np.array([1.5068182e+01, 6.7727275e+00, 4.4094244e-01, 6.8477272e+01,
        1.2823775e+02, 6.5737528e-01, 1.2552679e+03, 2.2174017e+00,
        5.9947405e+00, 1.6800790e+02, 4.0232184e+02, 2.2348366e+00,
        6.0216384e+00,
        0,0,0,0], dtype=np.float32)
            
                self.std= np.array([8.8430672e+00, 1.2588633e+00, 2.1731496e-01, 3.8057072e+01,
        8.1543999e+01, 3.7534222e-01, 1.0305146e+03, 2.6845067e+00,
        9.4314069e-01, 1.0874526e+02, 2.4056461e+02, 6.7858462e+00,
        1.6485872e+00, 
        1,1,1,1], dtype=np.float32)


            else:
                self.mean=np.array([1.5068182e+01, 6.7727275e+00, 4.4094244e-01, 6.8477272e+01,
        1.2823775e+02, 6.5737528e-01, 1.2552679e+03, 2.2174017e+00,
        5.9947405e+00, 1.6800790e+02, 4.0232184e+02, 2.2348366e+00,
        6.0216384e+00, 2.1604528e+00, 5.9982104e+00,0,0,0,0], dtype=np.float32)
            
                self.std= np.array([8.8430672e+00, 1.2588633e+00, 2.1731496e-01, 3.8057072e+01,
        8.1543999e+01, 3.7534222e-01, 1.0305146e+03, 2.6845067e+00,
        9.4314069e-01, 1.0874526e+02, 2.4056461e+02, 6.7858462e+00,
        1.6485872e+00, 2.6178448e+00, 9.2774606e-01,1,1,1,1], dtype=np.float32)

        elif self.name =='california_tomato':
            self.mean=np.array([1.52500000e+01, 6.50000000e+00, 3.15283656e-01, 6.03125000e+01,
                                3.07615387e+02, 5.10523260e-01, 4.98117859e+02, 1.24298476e-01,
                                6.86775160e+00, 1.21339521e+01, 4.01664978e+02, 0,0,0,0], dtype=np.float32)
        
            self.std= np.array([8.2120342e+00, 1.1180340e+00, 1.6626830e-01, 3.1970348e+01,
                                2.0187964e+02, 2.8188160e-01, 4.2335272e+02, 5.5605030e-01,
                                6.5448678e-01, 1.5673932e+01, 2.2597920e+02, 1,1,1,1], dtype=np.float32)

        # self.mean=0
        # self.std=1


        if self.observation_set in ['default',]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        elif self.observation_set in ['basic',]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        if self.observation_set in ['forecast',]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        
        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
    
        elif self.action_set=='depth_discreet':
            self.action_depths=[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
            self.action_space = spaces.Discrete(len(self.action_depths))    

        elif self.action_set=='binary':
            self.action_depths=[0,self.max_irr]
            self.action_space = spaces.Discrete(len(self.action_depths))    

            

                
    def states(self):
        return dict(type='float', shape=(self.observation_space.shape[0],))
 
    def actions(self):
        return dict(type='float', num_values=self.action_space.shape[0])
        
    def reset(self):
        """
        re/initialize model and return first observation
        """

        if not self.eval:

            sim_year=int(np.random.choice(np.arange(self.year1,self.year2+1)))
            self.wdf = self.gendf[self.gendf.simyear==sim_year].drop('simyear',axis=1)
            self.chosen=sim_year*1

        else:

            self.wdf = self.gendf[self.gendf.simyear==self.year1].drop('simyear',axis=1)
            self.chosen=self.year1*1

            
        month = self.planting_month
        day=self.planting_day
        self.model = AquaCropModel(f'{self.simcalyear}/{month}/{day}',f'{self.simcalyear}/12/31',
                                self.wdf,self.soil,self.crop,
                                IrrMngt=IrrMngtClass(IrrMethod=5),
                                InitWC=self.init_wc,CO2conc=self.CO2conc)
        self.model.initialize()

        if not self.include_rain:
            self.model.weather[:,2]=0

        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1)
            self.model.step(dayshift)
        
        self.irr_sched=[]

        return self.get_obs(self.model.InitCond)
 
    def get_obs(self,InitCond):
        """
        package the desired variables from InitCond into a numpy array
        and return as observation
        """

        # calculate relative depletion
        if InitCond.TAW>0:
            dep = InitCond.Depletion/InitCond.TAW
        else:
            dep=0

        # calculate mean daily precipitation and ETo from last 7 days
        start = max(0,self.model.ClockStruct.TimeStepCounter -7)
        end = self.model.ClockStruct.TimeStepCounter
        forecast1 = self.model.weather[start:end,2:4].mean(axis=0).flatten()

        # calculate sum of daily precipitation and ETo for whole season so far
        start2 = max(0,self.model.ClockStruct.TimeStepCounter -InitCond.DAP)
        forecastsum = self.model.weather[start2:end,2:4].sum(axis=0).flatten()

        #  yesterday precipitation and ETo and irr
        start2 = max(0,self.model.ClockStruct.TimeStepCounter-1)
        forecast_lag1 = self.model.weather[start2:end,2:4].flatten()

        # calculate mean daily precipitation and ETo for next N days
        start = self.model.ClockStruct.TimeStepCounter
        end = start+self.forecast_lead_time
        forecast2 = self.model.weather[start:end,2:4].mean(axis=0).flatten()
        
        # state 

        # month and day
        month = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).month
        day = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).day
        
        if self.observation_set in ['default','basic']:
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1,forecast2,]).flatten()


        gs = np.clip(int(self.model.InitCond.GrowthStage)-1,0,4)
        gs_1h = np.zeros(4)
        gs_1h[gs]=1

        # ir_sched = np.zeros(7)
        # for idx,ir in enumerate(reversed(self.irr_sched)):
        #     ir_sched[idx] = ir

        #     if idx==6:
        #         break

        if self.observation_set in ['default','forecast']:
            obs=np.array([
                        day,
                        month,
                        dep, # root-zone depletion
                        InitCond.DAP,#days after planting
                        InitCond.IrrCum, # irrigation used so far
                        InitCond.CC,
                        InitCond.B,
                        # gs,
                        # InitCond.GrowthStage,
                        
                        ]
                        +[f for f in forecast]
                        # +[ir for ir in ir_sched]
                        +[g for g in gs_1h]

                        , dtype=np.float32).reshape(-1)

        elif self.observation_set in ['basic']:

            obs=np.array([
                        day,
                        month,
                        dep, # root-zone depletion
                        InitCond.DAP,#days after planting
                        InitCond.IrrCum, # irrigation used so far
                        gs,
                        # InitCond.GrowthStage,
                        
                        ]
                        +[f for f in forecast]
                        # +[g for g in gs_1h]
                        # +[ir for ir in ir_sched]
                        , dtype=np.float32).reshape(-1)

        else:
            assert 1==2, 'no obs set'
        
        if self.normalize_obs:
            return (obs-self.mean)/self.std
        else:
            return obs
        
        
    def step(self,action):
        
        """
        take in binary action [   irrigation yes (1) or no (0)  ]
        and irrigate to field capacity or max irrigation
        """
        if self.action_set in ['depth_discreet']:

            depth = self.action_depths[int(action)]

            self.model.ParamStruct.IrrMngt.depth = depth

        elif self.action_set in ['binary']:

            if action == 1:
                # depth = np.clip(self.model.InitCond.Depletion,0,self.max_irr)
                depth = self.max_irr
            else:
                depth=0
            
            self.model.ParamStruct.IrrMngt.depth = depth


        elif self.action_set in ['depth']:

            depth=(action[0]+1)*12.5
#         self.model.ParamStruct.IrrMngt.depth = np.clip(depths[0],0,25)
            self.model.ParamStruct.IrrMngt.depth = depth


        elif self.action_set=='smt4':
            new_smt=np.ones(4)*(action+1)*50


        # new_smt=np.ones(4)*(action)*5
        # new_smt+=np.array([0.58709859, 0.66129679, 0.34608978, 0.10645481])*100

        start_day = self.model.InitCond.DAP
        start_y = self.model.InitCond.Y*1.
        start_irr = self.model.InitCond.IrrCum*1.

        for i in range(self.days_to_irr):
            
            if self.action_set in ['depth_discreet','binary','depth']:
                self.irr_sched.append(self.model.ParamStruct.IrrMngt.depth)
                self.model.step()
                self.model.ParamStruct.IrrMngt.depth = 0
            
            # self.model.ParamStruct.IrrMngt.depth = np.clip(depths[i],0,25)
            # self.model.step()
            # self.model.ParamStruct.IrrMngt.depth = 0
            # depth-=25


            elif self.action_set=='smt4':

                if self.model.InitCond.TAW>0:
                    dep = self.model.InitCond.Depletion/self.model.InitCond.TAW
                else:
                    dep=0

                gs = int(self.model.InitCond.GrowthStage)-1
                if gs<0 or gs>3:
                    depth=0
                else:
                    if 1-dep< (new_smt[gs])/100:
                        depth = np.clip(self.model.InitCond.Depletion,0,self.max_irr)
                    else:
                        depth=0
    
                self.model.ParamStruct.IrrMngt.depth = depth
                self.irr_sched.append(self.model.ParamStruct.IrrMngt.depth)

                self.model.step()

  
            # termination conditions
            if self.model.ClockStruct.ModelTermination:
                break

            now_day = self.model.InitCond.DAP
            if (now_day >0) and (now_day<start_day):
                break
 
        # step_reward = (self.CROP_PRICE*(max(self.model.InitCond.Y-start_y,0)) - self.IRRIGATION_COST*(max(self.model.InitCond.IrrCum-start_irr,0)))
        
        # - self.FIXED_COST )
 
        done = self.model.ClockStruct.ModelTermination
        
        reward = 0
 
        next_obs = self.get_obs(self.model.InitCond)
 
        if done:
        
            self.tsteps+=1

            # calculate profit 
            end_reward = (self.CROP_PRICE*self.model.Outputs.Final['Yield (tonne/ha)'].mean()
                        - self.IRRIGATION_COST*self.model.Outputs.Final['Seasonal irrigation (mm)'].mean()
                        - self.FIXED_COST )

            
            
            self.reward=end_reward
 
            rew = end_reward - self.best[self.chosen-1] 
            # rew = end_reward - global_best[self.chosen-1] 
            if rew>0:
                self.best[self.chosen-1]=end_reward
            if self.tsteps%100==0:
                self.total_best=self.best*1
                print(self.chosen,self.tsteps,self.best[:self.year2].mean())

            if self.eval:
                # print('yield',self.model.Outputs.Final['Yield (tonne/ha)'].mean())
                reward=end_reward*1000

            else:
                reward=end_reward
 
 
        return next_obs,reward/1000,done,dict()
 
 
    
    def get_mean_std(self,num_reps):
        """
        Function to get the mean and std of observations in an environment
 
        *Arguments:*
 
        `env`: `Env` : chosen environment
        `num_reps`: `int` : number of repetitions
 
        *Returns:*
 
        `mean`: `float` : mean of observations
        `std`: `float` : std of observations
 
        """
        self.mean=0
        self.std=1
        obs=[]
        for i in range(num_reps):
            self.reset()
 
            d=False
            while not d:
 
                ob,r,d,_=self.step(np.random.choice([0,1],p=[0.9,0.1]))
                # ob,r,d,_=self.step(-0.5)
                # ob,r,d,_=self.step(np.random.choice([-1.,0.],p=[0.9,0.1]))
                obs.append(ob)
 
        obs=np.vstack(obs)
 
        mean=obs.mean(axis=0)
 
        std=obs.std(axis=0)
        std[std==0]=1
 
        self.mean=mean
        self.std=std


import copy

class OptimCropEnv(gym.Env):

    def __init__(self, config):
        super(OptimCropEnv, self).__init__()


        self.config=copy.deepcopy(config)

        self.observation_set=config["observation_set"]
        self.normalize_obs = config["normalize_obs"]
        self.action_set=config["action_set"]


        if self.observation_set in ['default','forecast']:
                # with year sum

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        elif self.action_set=='depth_discreet':
            self.action_depths=[0,10,20,30,40,50]
            self.action_space = spaces.Discrete(len(self.action_depths))    

        elif self.action_set=='binary':
            self.action_depths=[0,self.max_irr]
            self.action_space = spaces.Discrete(len(self.action_depths))    

    def step(self, action):



        state, reward, yeardone, _ = self.eval_env.step(action)
        if yeardone and self.config['year1']==70:
            done=True
        elif yeardone:
            self.config['year1']+=1
            self.config['year2']+=1
            self.eval_env = CropEnv(self.config)
            state = self.eval_env.reset()

            done=False
        else:
            done=False


        return state, (reward/70)/1000, done, _


    def reset(self):

        self.config['year1']=1
        self.config['year2']=1
        self.config['evaluation_run']=True
        self.eval_env = CropEnv(self.config)

        self.reward=0

        obs = self.eval_env.reset()

        return obs
