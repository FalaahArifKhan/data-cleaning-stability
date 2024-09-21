# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "heart boost_clean mixed_exp boost_clean 1"
    "heart boost_clean mixed_exp boost_clean 2"
    "heart boost_clean mixed_exp boost_clean 3"
    "heart boost_clean mixed_exp boost_clean 4"
    "heart boost_clean mixed_exp boost_clean 5"
    "heart boost_clean mixed_exp boost_clean 6"
#     "heart deletion mixed_exp gandalf_clf 1"
#     "heart deletion mixed_exp gandalf_clf 2"
#     "heart deletion mixed_exp gandalf_clf 3"
#     "heart deletion mixed_exp gandalf_clf 4"
#     "heart deletion mixed_exp gandalf_clf 5"
#     "heart deletion mixed_exp gandalf_clf 6"
#     "heart median-mode mixed_exp gandalf_clf 1"
#     "heart median-mode mixed_exp gandalf_clf 2"
#     "heart median-mode mixed_exp gandalf_clf 3"
#     "heart median-mode mixed_exp gandalf_clf 4"
#     "heart median-mode mixed_exp gandalf_clf 5"
#     "heart median-mode mixed_exp gandalf_clf 6"
#     "heart median-dummy mixed_exp gandalf_clf 1"
#     "heart median-dummy mixed_exp gandalf_clf 2"
#     "heart median-dummy mixed_exp gandalf_clf 3"
#     "heart median-dummy mixed_exp gandalf_clf 4"
#     "heart median-dummy mixed_exp gandalf_clf 5"
#     "heart median-dummy mixed_exp gandalf_clf 6"
#     "heart miss_forest mixed_exp gandalf_clf 1"
#     "heart miss_forest mixed_exp gandalf_clf 2"
#     "heart miss_forest mixed_exp gandalf_clf 3"
#     "heart miss_forest mixed_exp gandalf_clf 4"
#     "heart miss_forest mixed_exp gandalf_clf 5"
#     "heart miss_forest mixed_exp gandalf_clf 6"
#     "heart k_means_clustering mixed_exp gandalf_clf 1"
#     "heart k_means_clustering mixed_exp gandalf_clf 2"
#     "heart k_means_clustering mixed_exp gandalf_clf 3"
#     "heart k_means_clustering mixed_exp gandalf_clf 4"
#     "heart k_means_clustering mixed_exp gandalf_clf 5"
#     "heart k_means_clustering mixed_exp gandalf_clf 6"
#     "heart automl mixed_exp gandalf_clf 1"
#     "heart automl mixed_exp gandalf_clf 2"
#     "heart automl mixed_exp gandalf_clf 3"
#     "heart automl mixed_exp gandalf_clf 4"
#     "heart automl mixed_exp gandalf_clf 5"
#     "heart automl mixed_exp gandalf_clf 6"
)

TEMPLATE_FILE="../cluster/run_mixed_exp/run-mixed-exp-models-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset null_imputer evaluation_scenario model run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_mixed_exp/sbatch_files/models/run-mixed-exp-${dataset}_${null_imputer}_${evaluation_scenario}_${model}_${index}.sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<DATASET>/${dataset}/g" -e "s/<EVALUATION_SCENARIO>/${evaluation_scenario}/g" -e "s/<NULL_IMPUTER>/${null_imputer}/g" -e "s/<MODEL>/${model}/g" -e "s/<RUN_NUMS>/${run_nums}/g" $TEMPLATE_FILE > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
