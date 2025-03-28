# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "heart datawig mixed_exp gandalf_clf 1"
    "heart datawig mixed_exp gandalf_clf 2"
    "heart datawig mixed_exp gandalf_clf 3"
    "heart datawig mixed_exp gandalf_clf 4"
    "heart datawig mixed_exp gandalf_clf 5"
    "heart datawig mixed_exp gandalf_clf 6"
)

TEMPLATE_FILE="../cluster/run_mixed_exp/run-mixed-exp-models-gpu-template.sbatch"

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
