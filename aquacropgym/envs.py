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
     

        self.best=config["best"]*1
        self.total_best=config["best"]*1
        self.tsteps=0
        
    
        self.observation_set=config["observation_set"]
        self.normalize_obs = config["normalize_obs"]
        self.action_set=config["action_set"]
        self.forecast_lead_time=config["forecast_lead_time"]

        if self.observation_set in ['default','forecast']:
                # with year sum
            self.mean=np.array([1.57299742e+01, 6.94973948e+00, 5.78204154e-01, 7.45930868e+01,
                5.67243665e+01, 4.45406029e-01, 8.44234541e+02, 2.44740967e+00,
                2.15764884e+00, 5.88546047e+00, 1.86397289e+02, 4.36919733e+02])
        
            self.std= np.array([8.64142443e+00, 1.36184203e+00, 2.26393775e-01, 4.20119480e+01,
                4.92215209e+01, 3.53966438e-01, 8.06799001e+02, 1.04439229e+00,
                2.64194311e+00, 9.92729799e-01, 1.17995847e+02, 2.59921570e+02])

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=float)
        
        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=float)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=float)
    
        elif self.action_set=='depth_discreet':
            self.action_depths=[0,10,20,30,40,50]
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

        else:

            self.wdf = self.gendf[self.gendf.simyear==self.year1].drop('simyear',axis=1)

            
        month = 5
        day=1
        self.model = AquaCropModel(f'{2018}/{month}/{day}',f'{2018}/12/31',
                                self.wdf,self.soil,self.crop,
                                IrrMngt=IrrMngtClass(IrrMethod=5),
                                InitWC=self.init_wc,)
        self.model.initialize()

        if not self.include_rain:
            self.model.weather[:,2]=0

        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1)
            self.model.step(dayshift)
        
 
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

        # calculate mean daily precipitation and ETo for next N days
        start = self.model.ClockStruct.TimeStepCounter
        end = start+self.forecast_lead_time
        forecast2 = self.model.weather[start:end,2:4].mean(axis=0).flatten()
        
        # month and day
        month = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).month
        day = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).day
        
        if self.observation_set=='default':
            forecast = np.concatenate([forecast1,forecastsum]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast2,forecastsum]).flatten()

        if self.observation_set in ['default','forecast']:
            obs=np.array([
                        day,
                        month,
                        dep, # root-zone depletion
                        InitCond.DAP,#days after planting
                        InitCond.IrrCum, # irrigation used so far
                        InitCond.CC,
                        InitCond.B,
                        InitCond.GrowthStage,
                        
                        ]
                        +[f for f in forecast]
                        )

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
        if self.action_set in ['depth_discreet','binary']:

            depth = self.action_depths[int(action)]

            self.model.ParamStruct.IrrMngt.depth = depth


#         depths=np.ones(7)*(action+1)*25
#         self.model.ParamStruct.IrrMngt.depth = np.clip(depths[0],0,25)


        elif self.action_set=='smt4':
            new_smt=np.ones(4)*(action+1)*50


        # new_smt=np.ones(4)*(action)*5
        # new_smt+=np.array([0.58709859, 0.66129679, 0.34608978, 0.10645481])*100

        start_day = self.model.InitCond.DAP
                
        for i in range(self.days_to_irr):
            
            if self.action_set in ['depth_discreet','binary']:

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
    
                self.model.step()
    
  
            # termination conditions
            if self.model.ClockStruct.ModelTermination:
                break

            now_day = self.model.InitCond.DAP
            if (now_day >0) and (now_day<start_day):
                break
 
 
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
 
            rew = end_reward - self.total_best[self.chosen-1] 
            # rew = end_reward - global_best[self.chosen-1] 
            if rew>0:
                self.best[self.chosen-1]=end_reward
            if self.tsteps%100==0:
                self.total_best=self.best*1
                print(self.tsteps,self.best[:self.year2].mean())

            if self.eval:
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

