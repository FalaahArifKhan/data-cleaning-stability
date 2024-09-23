# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
#     "folk boost_clean mixed_exp lr_clf 1"
#     "folk boost_clean mixed_exp lr_clf 2"
#     "folk boost_clean mixed_exp lr_clf 3"
#     "folk boost_clean mixed_exp lr_clf 4"
#     "folk boost_clean mixed_exp lr_clf 5"
#     "folk boost_clean mixed_exp lr_clf 6"

#     "folk deletion mixed_exp gandalf_clf 1"
#     "folk deletion mixed_exp gandalf_clf 2"
#     "folk deletion mixed_exp gandalf_clf 3"
#     "folk deletion mixed_exp gandalf_clf 4"
#     "folk deletion mixed_exp gandalf_clf 5"
#     "folk deletion mixed_exp gandalf_clf 6"
#     "folk median-mode mixed_exp gandalf_clf 1"
#     "folk median-mode mixed_exp gandalf_clf 2"
#     "folk median-mode mixed_exp gandalf_clf 3"
#     "folk median-mode mixed_exp gandalf_clf 4"
#     "folk median-mode mixed_exp gandalf_clf 5"
#     "folk median-mode mixed_exp gandalf_clf 6"
#     "folk median-dummy mixed_exp gandalf_clf 1"
#     "folk median-dummy mixed_exp gandalf_clf 2"
#     "folk median-dummy mixed_exp gandalf_clf 3"
#     "folk median-dummy mixed_exp gandalf_clf 4"
#     "folk median-dummy mixed_exp gandalf_clf 5"
#     "folk median-dummy mixed_exp gandalf_clf 6"

#     "folk miss_forest mixed_exp gandalf_clf 1"
#     "folk miss_forest mixed_exp gandalf_clf 2"
#     "folk miss_forest mixed_exp gandalf_clf 3"
#     "folk miss_forest mixed_exp gandalf_clf 4"
#     "folk miss_forest mixed_exp gandalf_clf 5"
#     "folk miss_forest mixed_exp gandalf_clf 6"
    "folk k_means_clustering mixed_exp gandalf_clf 1"
    "folk k_means_clustering mixed_exp gandalf_clf 2"
    "folk k_means_clustering mixed_exp gandalf_clf 3"
    "folk k_means_clustering mixed_exp gandalf_clf 4"
    "folk k_means_clustering mixed_exp gandalf_clf 5"
    "folk k_means_clustering mixed_exp gandalf_clf 6"
#     "folk automl mixed_exp gandalf_clf 1"
#     "folk automl mixed_exp gandalf_clf 2"
#     "folk automl mixed_exp gandalf_clf 3"
#     "folk automl mixed_exp gandalf_clf 4"
#     "folk automl mixed_exp gandalf_clf 5"
#     "folk automl mixed_exp gandalf_clf 6"
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
