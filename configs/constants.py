from enum import Enum


EXPERIMENT_RUN_SEEDS = [100 * i for i in range(1, 11)]
NUM_FOLDS_FOR_TUNING = 3

# ====================================================================
# Datasets
# ====================================================================
GERMAN_CREDIT_DATASET = "german"
BANK_MARKETING_DATASET = "bank"
CARDIOVASCULAR_DISEASE_DATASET = "heart"
DIABETES_DATASET = "diabetes"
LAW_SCHOOL_DATASET = "law_school"
ACS_INCOME_DATASET = "folk"


# ====================================================================
# Error Injection Strategies
# ====================================================================
class ErrorInjectionStrategy(Enum):
    mcar = 'MCAR'
    mar = 'MAR'
    mnar = 'MNAR'

    def __str__(self):
        return self.value


# ====================================================================
# Error Repair Methods
# ====================================================================
class ErrorRepairMethod(Enum):
    deletion = 'deletion'
    median_mode = 'median-mode'
    median_dummy = 'median-dummy'
    miss_forest = 'missForest'
    k_means_clustering = 'kMeansClustering'
    datawig = 'datawig'
    discriminative_dl = 'discriminativeDL'
    boost_clean = 'boostClean'
    cp_clean = 'cpClean'

    def __str__(self):
        return self.value


# ====================================================================
# ML Models
# ====================================================================
class MLModels(Enum):
    lr_clf = 'lr_clf'
    dt_clf = 'dt_clf'
    lgbm_clf = 'lgbm_clf'
    nn_clf = 'nn_clf'

    def __str__(self):
        return self.value
