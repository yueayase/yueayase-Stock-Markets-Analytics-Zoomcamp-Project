from scripts.data_repo import DataRepository
from scripts.transform import TransformData
from scripts.train import TrainModel
from scripts.simulation import Simulation

import warnings

def main():
    FETCH_REPO = False
    TRANSFORM_DATA = False
    TRAIN_MODEL = False

    print('Step 1: Getting data from APIs or Load from disk')
    repo = DataRepository()

    if FETCH_REPO:
        # Fetch All 3 datasets for all dates from APIs
        repo.fetch()
        # save data to a local dir
        repo.persist(data_dir='local_data/')
    else:
        # OR Load from disk
        repo.load(data_dir='local_data/')
    
    print('Step 2: Making data transformations (combining into one dataset)')

    transformed =  TransformData(repo = repo)

    if TRANSFORM_DATA:
        transformed.transform()
        transformed.persist(data_dir='local_data/')
    else:
        transformed.load(data_dir='local_data/')

    print('Step 3: Training the model or loading from disk')

    # Suppress all warnings (not recommended in production unless necessary)
    warnings.filterwarnings("ignore")
    
    trained = TrainModel(transformed=transformed)
    trained.prepare_dataframe() # prepare dataframes
    trained.train_model(train_new=TRAIN_MODEL, tuning_rf=False) # train or load the model
    trained.make_inference()    # produce some correctness analysis data used in the future

    print('Step 4: Simulation')
    sim = Simulation(train_model=trained, invest_each=100.0)
    sim.report("Sim1: Portfolio Strategy, Equal Weights", opt_method="EW")
    sim.report("Sim2: Portfolio Strategy, Mean-Variance Optimization", opt_method="MVO")


if __name__ == '__main__':
    main()