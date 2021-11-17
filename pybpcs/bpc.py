from warnings import warn
import math
import numpy as np
import pandas as pd
from typing import Union

class bpc():
    def __init__(self, data: pd.DataFrame, player0: str, player1: str, model_type: str = 'bt', result_column: Union(str, None) = None, player0_score: Union(str, None)=None, player1_score: Union(str, None)=None, z_player1: Union(str, None) = None, cluster: Union(str, None) = None, predictors=None, subject_predictors = None, solve_ties:str = 'random', win_score:str = 'higher', priors = None, chains:int = 4, parallel_chains:int = 4, iter:int = 2000, warmup:int = 1000, show_chain_messages:bool = False, seed = None, log_lik:bool=True, dir: Union(str, None)=None):
        # first checks if model is consistent
        if ((player0_score is None) or (player1_score is None)) and (result_column is None):
            raise ValueError( 'Error! It is required to have either scores for both player0 and player1 OR a column indicating who won (0 for player0 and 1 for player1)')
        
        #TODO: check if data is of pandas type

        if (z_player1 is not None) and ('-ordereffect' not in model_type):
            raise ValueError('Error! If the order effect column is specified you should choose a model with ordereffect')

        if (cluster is not None) and ('-U' not in model_type):
            raise ValueError('Error! If the cluster column is specified you should choose a model to handle the random effects of the cluster')


        if (predictors is not None) and ('-generalized' not in model_type):
            raise ValueError('Error! If the predictors dataframe is specified you should choose a generalized model')

        if (solve_ties != 'none') and ('davidson' in model_type):
            warn('You are calling a variation of the Davidson model but you are handling the ties. Consider switching to a Bradley-Terry model or setting solve_ties to none')
 


        # saving the input arguments
        self._data = data
        self._player0 = player0
        self._player1 = player1
        self._player0_score = player0_score
        self._player1_score = player1_score
        self._result_column = result_column
        self._z_player1 = z_player1
        self._cluster = cluster
        self._predictors = predictors
        self._model_type = model_type
        self._solve_ties = solve_ties
        self._subject_predictors = subject_predictors
        self._win_score = win_score
        self._priors = priors
        self._chains = chains
        self._iter = iter
        self._warmup = warmup
        self._seed = seed

        #Rate of messages in Stan
        self._refresh = False
        if show_chain_messages is False:
            self._refresh = 0
        else:
             self._refresh = math.floor(self._iter / 10)

        #Now fix the data frame and the parameters so it can be used by cmdstanpy

        ## lets first drop na since we are not handling those
        data_cols = list(self._data.columns)
        dropna_cols = []
        for z in [self._player0, self._player1, self._player0_score, self._player1_score, self._result_column, self._subject_predictors, self._z_player1]:
            if z is not None:
                if z in data_cols:
                    dropna_cols.append(z)
        
        self._data.dropna(subset=dropna_cols, inplace=True)

        # check the values of the result column and give an error in case it does not fulfill the requirementss
        results_values = list(self._data[self._result_column].values)
        for v in results_values:
            if v not in [0,1,2]:
                raise ValueError('The result column should only contain values 0, 1 or 2')
        
        # We fix the ties and compute a result column from the scores
        self._compute_scores_ties()
        # if we have ties, and we are not using the davidson model and we are not solving the ties we have an error
        if (2 in self._data[self._result_column].values) and ('davidson' not in self._model_type) and (self._solve_ties == 'none'):
            raise ValueError('We see ties on the data. If not handling ties a version of the Davidson model should be used ')
       
        # We check the order effect column to see if contains values different than 0 or 1 
        if(self._z_player1 is not None):
            if self._data[self._z_player1].values != 1 and  self._data[self._z_player1].values != 0:
                raise ValueError('The z_player1 column should contain only 0 and 1.')

        # Now that we created the basic checks we need to handle the factors and the different conditions
        # For the Stan model we need the indexes of the players etc and not the actual names
        # So we will create some index and lookup tables
        
        self._create_index_columns()
        
        #TODO: add subject predictors matrix and lookup table

        #TODO: add generalized model predictors and lookuptable

        #TODO: add cluster lookup table

        #Default priors
        default_std = 3.0
        default_mu = 3.0

        prior_lambda_std = None
        prior_lambda_mu = None
        prior_nu_std = None
        prior_nu_mu = None
        prior_gm_mu = None
        prior_gm_std = None
        prior_U1_std = None
        prior_U2_std = None
        prior_U3_std = None
        prior_S_std = None
        
        #Setting up custom priors

        
        
        pass


    def sample(self):
        pass

    def predict(self):
        pass

    def summary(self):
        pass

    def parameters(self):
        pass

    def probability_table(self):
        pass

    def posterior(self):
        pass

    def waic(self):
        pass

    def plot(self):
        pass


    def _compute_scores_ties(self):
        # If we have the scores we need to compute the result that will be in column y
        if self._player0_score is not None and self._player1_score is not None:        
            self._data['diff_score'] = self._data[self._player1_score] - self._data[self._player0_score]
            if self._win_score == 'higher':
                self._data.loc[self._data['diff_score'] > 0 , self._result_column ] = 1
                self._data.loc[self._data['diff_score'] < 0 , self._result_column ] = 0
                self._data.loc[self._data['diff_score'] == 0 , self._result_column ] = 2
            if self._win_score == 'lower':
                self._data.loc[self._data['diff_score'] < 0 , self._result_column ] = 1
                self._data.loc[self._data['diff_score'] > 0 , self._result_column ] = 0
                self._data.loc[self._data['diff_score'] == 0 , self._result_column ] = 2
        
        # Now let's fix the ties
        if self._solve_ties != 'none':
            if self._solve_ties == 'random':
                for index, row in self._data.iterrows():
                    if row[self._result_column] == 2:
                        row[self._result_column] = np.random.randint(2)
            if self._solve_ties == 'remove':
                self._data = self._data[self._data[self._result_column]!=2]        

        pass


    def _create_index_lookup_table(self):
        #Stan is 1 indexed

        # Player lookuptable
        names = np.unique(self._data[[self._player0, self._player1]].values)
        index = np.linspace(start=1, stop=names.size,num=names.size)
        self._lookuptable = pd.DataFrame({'Names': names, 'Index': index})
        self._lookuptable['Index'] = self._lookuptable['Index'].astype(int)
       
        pass
    
    def _create_index_columns(self):
        # We first need to create a lookup table
        self._create_index_lookup_table()
    
        # https://stackoverflow.com/questions/35469399/pandas-table-lookup
        def lookup_index(name):
            match = self._lookuptable['Names'] == name #filter which to apply based on the name
            index = self._lookuptable['Index'][match].values[0] # get the value of the index
            return index
        self._data['player0_index'] = self._data[self._player0].apply(lookup_index)
        self._data['player1_index'] = self._data[self._player1].apply(lookup_index)
        pass