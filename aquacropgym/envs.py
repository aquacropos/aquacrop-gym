from aquacrop.classes import *
from aquacrop.core import *
 
import gym
from gym import spaces

import numpy as np



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
    max_irr_season=10_000, # maximum irrigation appl for season
    irr_cap_half_DAP=-999, # day after planting to half water supply
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


class CropEnv(gym.Env):
    """
    AquaCrop-OSPy crop environment in openai-gym style.

    Cropping system consists of a single crop on 1 HA of homogenous soil.
    Each episode consits of 1 season of this system

    Config parameters will specify the type of cropping environment.

    Every `days_to_irr` days, the agent will see an observation of the enviornemnt

    The agent will then make an irrigation decision which is passed to the environment

    This proicess continues until season has finished

    the seasonal profit is calculated and apssed to the agent as the reward
    """
 
    def __init__(self,config):

        
        super(CropEnv, self).__init__()
 
        self.gendf = config["gendf"]
        self.days_to_irr=config["days_to_irr"]
        self.eval= config['evaluation_run']
        self.year1=config["year1"]
        self.year2=config["year2"] 
        self.dayshift = config["dayshift"]
        self.include_rain=config["include_rain"]
        self.max_irr=config["max_irr"]
        self.init_wc = config["init_wc"]
        self.CROP_PRICE=config["crop_price"]
        self.IRRIGATION_COST=config["irrigation_cost"] 
        self.FIXED_COST = config["fixed_cost"]
        self.planting_month = int(config['planting_date'].split('/')[0])
        self.planting_day = int(config['planting_date'].split('/')[1])
        self.max_irr_season=config['max_irr_season']
        self.irr_cap_half_DAP=config['irr_cap_half_DAP']
        self.name=config["name"]
        self.best=config["best"]*1
        self.total_best=config["best"]*1
        self.simcalyear=config["simcalyear"]
        self.CO2conc=config["CO2conc"]
        self.observation_set=config["observation_set"]
        self.normalize_obs = config["normalize_obs"]
        self.action_set=config["action_set"]
        self.forecast_lead_time=config["forecast_lead_time"]

        # randomly drawn weather year

        if self.eval:
            self.chosen=self.year1
        else:
            self.chosen = np.random.choice([i for i in range(self.year1,self.year2+1)])

        # crop and soil choice

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
     
     
        self.tsteps=0

        # observation normalization

        self.mean=0
        self.std=1

        # obsservation and action sets

        if self.observation_set in ['default',]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-5., high=self.max_irr+5., shape=(1,), dtype=np.float32)
    
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
        re-initialize model and return first observation
        """

        if not self.eval:

            # choose a random training year to simulate

            sim_year=int(np.random.choice(np.arange(self.year1,self.year2+1)))
            self.wdf = self.gendf[self.gendf.simyear==sim_year].drop('simyear',axis=1)
            self.chosen=sim_year*1

        else:

            # simulate the specified year in in evaluation mode

            self.wdf = self.gendf[self.gendf.simyear==self.year1].drop('simyear',axis=1)
            self.chosen=self.year1*1

        # irrigation cap

        if isinstance(self.max_irr_season, list):
            if isinstance(self.max_irr_season[0], list):
                self.chosen_max_irr_season = float(np.random.choice(self.max_irr_season[0]))
            else:
                self.chosen_max_irr_season = float(np.random.randint(self.max_irr_season[0],self.max_irr_season[1]))
        else:
            self.chosen_max_irr_season =self.max_irr_season*1.
            

        # create and initialize model

        month = self.planting_month
        day=self.planting_day

        self.model = AquaCropModel(f'{self.simcalyear}/{month}/{day}',f'{self.simcalyear}/12/31',
                                self.wdf,self.soil,self.crop,
                                IrrMngt=IrrMngtClass(IrrMethod=5,MaxIrrSeason=self.chosen_max_irr_season),
                                InitWC=self.init_wc,CO2conc=self.CO2conc)
        self.model.initialize()

        # remove rainfall from weather if requested

        if not self.include_rain:
            self.model.weather[:,2]=0

        # shift the start day of simulation by specified amound
        # default 1

        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1)
            self.model.step(dayshift)
        
        # store irrigation events
        self.irr_sched=[]

        return self.get_obs(self.model.InitCond)
 
    def get_obs(self,InitCond):
        """
        package variables from InitCond into a numpy array
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
        
        # concatenate all weather variables

        if self.observation_set in ['default']:
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1,forecast2,]).flatten()

        # put growth stage in one-hot format

        gs = np.clip(int(self.model.InitCond.GrowthStage)-1,0,4)
        gs_1h = np.zeros(4)
        gs_1h[gs]=1

        # create observation array

        if self.observation_set in ['default','forecast']:
            obs=np.array([
                        day,
                        month,
                        dep, # root-zone depletion
                        InitCond.DAP,#days after planting
                        InitCond.IrrCum, # irrigation used so far
                        InitCond.CC,
                        InitCond.B,
                        self.chosen_max_irr_season-InitCond.IrrCum,
                        # InitCond.GrowthStage,
                        
                        ]
                        +[f for f in forecast]
                        # +[ir for ir in ir_sched]
                        +[g for g in gs_1h]

                        , dtype=np.float32).reshape(-1)


        else:
            assert 1==2, 'no obs set'
        
        if self.normalize_obs:
            return (obs-self.mean)/self.std
        else:
            return obs
        
    def step(self,action):
        """
        Take in agents action choice

        apply irrigation depth on following day

        simulate N days until next irrigation decision point

        calculate and return profit at end of season

        """

        # if choosing discrete depths

        if self.action_set in ['depth_discreet']:

            depth = self.action_depths[int(action)]

            self.model.ParamStruct.IrrMngt.depth = depth

        # if making banry yes/no irrigation decisions

        elif self.action_set in ['binary']:

            if action == 1:
                depth = self.max_irr #apply max irr
            else:
                depth=0
            
            self.model.ParamStruct.IrrMngt.depth = depth

        # if spefiying depth from continuous range

        elif self.action_set in ['depth']:

            depth=np.clip(action[0],0,self.max_irr)
            self.model.ParamStruct.IrrMngt.depth = depth

        # if deciding on soil-moisture targets

        elif self.action_set=='smt4':

            new_smt=np.ones(4)*(action+1)*50


        start_day = self.model.InitCond.DAP

        for i in range(self.days_to_irr):

            # apply depth next day, and no more events till next decision
            
            if self.action_set in ['depth_discreet','binary','depth']:
                self.irr_sched.append(self.model.ParamStruct.IrrMngt.depth)
                self.model.step()
                self.model.ParamStruct.IrrMngt.depth = 0
            
            # if specifying soil-moisture target, 
            # irrigate if root zone soil moisture content
            # drops below threshold

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

            if self.model.ClockStruct.ModelTermination is True:
                break

            now_day = self.model.InitCond.DAP
            if (now_day >0) and (now_day<start_day):
                # end of season
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
 
            # keep track of best rewards in each season
            rew = end_reward - self.best[self.chosen-1] 
            if rew>0:
                self.best[self.chosen-1]=end_reward
            if self.tsteps%100==0:
                self.total_best=self.best*1
                # print(self.chosen,self.tsteps,self.best[:self.year2].mean())

            # scale reward
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