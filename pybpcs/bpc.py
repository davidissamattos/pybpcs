from warnings import warn
import math
import numpy as np
import pandas as pd
from typing import List, Union
from cmdstanpy import CmdStanModel
from pandas.core.frame import DataFrame

class bpc():
    def __init__(self, data: pd.DataFrame, player0: str, player1: str, model_type: str = 'bt', result_column: Union[str, None] = None, player0_score: Union[str, None]=None, player1_score: Union[str, None]=None, z_player1: Union[str, None] = None, cluster: Union[List[str], None] = None, predictors: Union[DataFrame, None]=None, subject_predictors: Union[List[str],None] =None, solve_ties:Union[str, None] = 'random', win_score:str= 'higher', priors: Union[dict, None] = None, chains:int = 4, parallel_chains:int = 4, iter:int = 2000, warmup:int = 1000, show_chain_messages:bool = False, seed = None, log_lik:bool=True, dir: Union[str, None]=None):
        
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
        self._parallel_chains = parallel_chains
        self._iter = iter
        self._warmup = warmup
        self._seed = seed
        self._log_lik = log_lik
        self._dir = dir
        
        
        # first checks if model is consistent
        if ((player0_score is None) or (player1_score is None)) and (result_column is None):
            raise ValueError( 'Error! It is required to have either scores for both player0 and player1 OR a column indicating who won (0 for player0 and 1 for player1)')
        
        # check the values of the result column and give an error in case it does not fulfill the requirementss
        results_values = list(self._data[self._result_column].values)
        for v in results_values:
            if v not in [0,1,2]:
                raise ValueError('The result column should only contain values 0, 1 or 2')


        #check if data is of pandas type
        if type(data) is not DataFrame:
            raise ValueError( 'Input data should be a Pandas DataFrame object')

        #order effect checks
        if (z_player1 is not None) and ('-ordereffect' not in model_type):
            raise ValueError('Error! If the order effect column is specified you should choose a model with ordereffect')
        if (z_player1 is None) and ('-ordereffect' in model_type):
            raise ValueError('Error! You need to provide a column indicating if player 1 had an order effect advantage or not. Use argument z_player1.')
        # We check the order effect column to see if contains values different than 0 or 1 
        if(self._z_player1 is not None):
            values = self._data[self._z_player1].values
            for v in results_values:
                if v not in [0,1]:
                    raise ValueError('The z_player1 column should contain only 0 and 1.')


        # random effects checks
        if (cluster is not None) and ('-U' not in model_type):
            raise ValueError('Error! If the cluster column is specified you should choose a model to handle the random effects of the cluster')
        if (cluster is None) and ('-U' in model_type):
            raise ValueError('Error! You need to provide column(s) in the argument cluster to fit random effects')
        if (cluster is not None) and (len(cluster)>3):
            raise ValueError('Error! You should add a maximum of 3 clusters only')

        #generalized models checks
        if (predictors is not None) and ('-generalized' not in model_type):
            raise ValueError('Error! If the predictors dataframe is specified you should choose a generalized model')
        if (predictors is None) and ('-generalized'  in model_type):
            raise ValueError('Error! You need to provide a data frame of predictors to use the generalized model')
                #check if data is of pandas type
        if type(predictors) is not DataFrame:
            raise ValueError( 'The predictors should be a Pandas DataFrame object')


        #subject predictors
        if (subject_predictors is not None) and ('-subjectpredictors' not in model_type):
            raise ValueError('Error! If the subject_predictors dataframe is specified you should choose a subjectpredictors model')
        if (subject_predictors is None) and ('-subjectpredictors'  in model_type):
            raise ValueError('Error! You need to provide a data frame of subject_predictors to use the subjectpredictors model')

        # davidson models checks
        if (solve_ties != 'none') and (model_type.startswith('davidson')):
            warn('You are calling a variation of the Davidson model but you are handling the ties. Consider switching to a Bradley-Terry model or setting solve_ties to none')
        # We fix the ties and compute a result column from the scores
        self._compute_scores_ties()
        # if we have ties, and we are not using the davidson model and we are not solving the ties we have an error
        if (2 in self._data[self._result_column].values) and ('davidson' not in self._model_type) and (self._solve_ties == 'none'):
            raise ValueError('We see ties on the data. If not handling ties a version of the Davidson model should be used ')
       

        #TODO: subject predictors checks


        # lets first drop NA cases since we are not handling those
        data_cols = list(self._data.columns)
        dropna_cols = []
        for z in [self._player0, self._player1, self._player0_score, self._player1_score, self._result_column, self._subject_predictors, self._z_player1]:
            if z is not None:
                if z in data_cols:
                    dropna_cols.append(z)    
        self._data.dropna(subset=dropna_cols, inplace=True)


        #Rate of messages in Stan
        self._refresh = False
        if show_chain_messages is False:
            self._refresh = 0
        else:
             self._refresh = math.floor(self._iter / 10)


        # Now that we created the basic checks we need to handle the factors and the different conditions
        # For the Stan model we need the indexes of the players etc and not the actual names
        # So we will create some index and lookup tables
        self._create_lookuptables_and_index_columns()


        #Default priors
        default_std: float = 3.0
        default_mu: float = 0.0

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
        
        # Setting up custom priors
        if self._priors is not None:
            #lambda
            if 'prior_lambda_std' in self._priors.key():
                prior_lambda_std = self._priors['prior_lambda_std']
            else:
                prior_lambda_std = default_std
            if 'prior_lambda_mu' in self._priors.key():
                prior_lambda_mu = self._priors['prior_lambda_mu']
            else:
                prior_lambda_mu = default_mu
            #nu
            if 'prior_nu_std' in self._priors.key():
                prior_nu_std = self._priors['prior_nu_std']
            else:
                prior_nu_std = default_std
            if 'prior_nu_mu' in self._priors.key():
                prior_nu_mu = self._priors['prior_nu_mu']
            else:
                prior_nu_mu = default_mu
            #gm
            if 'prior_gm_std' in self._priors.key():
                prior_gm_std = self._priors['prior_gm_std']
            else:
                prior_gm_std = default_std
            if 'prior_gm_mu' in self._priors.key():
                prior_gm_mu = self._priors['prior_gm_mu']
            else:
                prior_gm_mu = default_mu
            #U1
            if 'prior_U1_std' in self._priors.key():
                prior_U1_std = self._priors['prior_U1_std']
            else:
                prior_U1_std = default_std
            #U2
            if 'prior_U2_std' in self._priors.key():
                prior_U2_std = self._priors['prior_U2_std']
            else:
                prior_U2_std = default_std
            #U3
            if 'prior_U3_std' in self._priors.key():
                prior_U3_std = self._priors['prior_U3_std']
            else:
                prior_U3_std = default_std
            #S
            if 'prior_S_std' in self._priors.key():
                prior_S_std = self._priors['prior_S_std']
            else:
                prior_S_std = default_std
        
        #TODO: setup standata based on the model_type
        # Set variable used to calculate the log_lik in stan
        calc_log_lik: int
        if self._log_lik is True:
            calc_log_lik = int(1)
        else:
            calc_log_lik = int(0)

        #TODO: basic data
        self._standata = {
            'y': self._data[self._result_column],
            'N_total': int(self._data.size),
            'N_players': int(self._players_lookuptable.size),
            'player0_indexes': self._data['player0_index'],
            'player1_indexes': self._data['player1_index'],
            'prior_lambda_std': prior_lambda_std,
            'prior_lambda_mu': prior_lambda_mu,
            'prior_gm_std': prior_gm_std,
            'prior_gm_mu': prior_gm_mu,
            'prior_nu_std': prior_nu_std,
            'prior_nu_mu': prior_nu_mu,
            'prior_U1_std': prior_U1_std,
            'prior_U2_std': prior_U2_std,
            'prior_U3_std': prior_U3_std,
            'prior_S_std': prior_S_std,
            'calc_log_lik': calc_log_lik
        }

        # now we condition on the type of model we set
        # We do this by setting some flags on the stan data and saving a vector with the name of the parameters we use the most
        self._used_pars = ['lambda']

        #Order effects
        if '-ordereffect' in self._model_type:
            self._standata['use_Ordereffect'] = 1
            self._standata['z_player1'] = self._data[self._z_player1]
            self._used_pars.append('gm')
        else:
            self._standata['use_Ordereffect'] = 0
            self._standata['z_player1'] = []

        #Random effects
        if '-U' in self._model_type:
            #A single cluster
            if len(self._cluster) == 1:
                #U1
                self._standata['use_U1'] = 1
                self._standata['N_U1'] = self._cluster_lookup_tables[0].size
                self._standata['U1_indexes'] = self._data['cluster1_index']
                #U2
                self._standata['use_U2'] = 0
                self._standata['N_U2'] = 0
                self._standata['U2_indexes'] = []
                #U3
                self._standata['use_U3'] = 0
                self._standata['N_U3'] = 0
                self._standata['U3_indexes'] = []
                 #used parameters
                self._used_pars.append('U1')
                self._used_pars.append('U1_std')
            elif len(self._cluster) == 2:
                #U1
                self._standata['use_U1'] = 1
                self._standata['N_U1'] = self._cluster_lookup_tables[0].size
                self._standata['U1_indexes'] = self._data['cluster1_index']
                #U2
                self._standata['use_U2'] = 1
                self._standata['N_U2'] = self._cluster_lookup_tables[1].size
                self._standata['U2_indexes'] = self._data['cluster2_index']
                #U3
                self._standata['use_U3'] = 0
                self._standata['N_U3'] = 0
                self._standata['U3_indexes'] = []
                #used parameters
                self._used_pars.append('U1')
                self._used_pars.append('U1_std')
                self._used_pars.append('U2')
                self._used_pars.append('U2_std')

            elif len(self._cluster) == 3:
                #U1
                self._standata['use_U1'] = 1
                self._standata['N_U1'] = self._cluster_lookup_tables[0].size
                self._standata['U1_indexes'] = self._data['cluster1_index']
                #U2
                self._standata['use_U2'] = 1
                self._standata['N_U2'] = self._cluster_lookup_tables[1].size
                self._standata['U2_indexes'] = self._data['cluster2_index']
                #U3
                self._standata['use_U3'] = 1
                self._standata['N_U3'] = self._cluster_lookup_tables[2].size
                self._standata['U3_indexes'] = self._data['cluster3_index']
                #used parameters
                self._used_pars.append('U1')
                self._used_pars.append('U1_std')
                self._used_pars.append('U2')
                self._used_pars.append('U2_std')
                self._used_pars.append('U3')
                self._used_pars.append('U3_std')
            else:
                #U1
                self._standata['use_U1'] = 0
                self._standata['N_U1'] = 0
                self._standata['U1_indexes'] = []
                #U2
                self._standata['use_U2'] = 0
                self._standata['N_U2'] = 0
                self._standata['U2_indexes'] = []
                #U3
                self._standata['use_U3'] = 0
                self._standata['N_U3'] = 0
                self._standata['U3_indexes'] = []

        #Subject predictors
        if '-subjectpredictors' in self._model_type:
            self._standata['use_SubjectPredictors'] = 1
            self._standata['N_SubjectPredictors'] = self._subject_predictors.size
            self._standata['X_subject'] = self._subject_predictors.as_matrix
            self._used_pars.append('S')
        else:
           self._standata['use_SubjectPredictors'] = 0 

        #Generalized models 
        if '-generalized' in self._model_type:
            self._standata['use_Generalized'] = 1
            self._standata['K'] = self._generalized_predictors_lookuptable.size
            #TODO: fix matrix X
            # self._standata['X'] = 
            self._used_pars.append('B')
        else:
            self._standata['use_Generalized'] = 0
            self._standata['K'] = 0
            self._standata['X'] = np.empty(shape=(0,0))
            

        #Davidson model
        if self._model_type.startswith('davidson'):
            self._standata['use_Davidson'] = 1
            self._used_pars.append('nu')
        else:
            self._standata['use_Davidson'] = 0


        pass


    def sample(self):
        # model = CmdStanModel(stan_file=stan_file)
        # self._fit = model.fit()
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
        if self._solve_ties is None:
            self._solve_ties = 'none'
        if self._solve_ties != 'none':
            if self._solve_ties == 'random':
                for index, row in self._data.iterrows():
                    if row[self._result_column] == 2:
                        row[self._result_column] = np.random.randint(2)
            if self._solve_ties == 'remove':
                self._data = self._data[self._data[self._result_column]!=2]        

        pass


    
    def _create_lookuptables_and_index_columns(self):
        # https://stackoverflow.com/questions/35469399/pandas-table-lookup
        def lookup_index(name, lookuptable):
            match = lookuptable['Names'] == name #filter which to apply based on the name
            index = lookuptable['Index'][match].values[0] # get the value of the index
            return index

        def create_index(df, col, lookuptable):
            return df[col].apply(lookup_index, lookuptable=lookuptable)

        def create_lookup_table(df, cols:List):
            #Stan is 1 indexed
            if len(cols)==1:
                cols = cols[0]
            names = np.unique(df[cols].values)
            index = np.linspace(start=1, stop=names.size,num=names.size)
            lookuptable = pd.DataFrame({'Names': names, 'Index': index})
            lookuptable['Index'] = lookuptable['Index'].astype(int)
            return lookuptable


        # We first need to create a lookup table for the players
        cols = [self._player0, self._player1]
        self._players_lookuptable = create_lookup_table(self._data, cols=cols)
        self._data['player0_index'] = create_index(self._data, self._player0, self._players_lookuptable)
        self._data['player1_index'] = create_index(self._data, self._player1, self._players_lookuptable)
        # self._data['player0_index'] = self._data[self._player0].apply(lookup_index, lookuptable=self._players_lookuptable)
        # self._data['player1_index'] = self._data[self._player1].apply(lookup_index, lookuptable=self._players_lookuptable)
        
        #TODO: handle other tables
        if self._predictors is not None:
            pass

        if self._cluster is not None:
            self._create_cluster_lookuptable = []
            for idx, cl in enumerate(self._cluster):
                lookuptable = self._create_lookup_table(self._data, cols=cl)
                self._cluster_lookup_tables.append(lookuptable)
                colname = 'cluster'+str(idx+1)+'_index'
                self._data[colname] = create_index(self._data, cl, lookuptable)
        pass