# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "law_school \'deletion\',\'median-mode\',\'median-dummy\' \'exp1_mcar3\' \'dt_clf\' 1,2,3,4,5,6"
)

TEMPLATE_FILE="../cluster/run_exp1/run-exp1-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset null_imputers evaluation_scenario model run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_exp1/sbatch_files/run-exp1-${dataset}_${null_imputers}_${evaluation_scenario}_${model}_${index}.sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<DATASET>/${dataset}/g" -e "s/<NULL_IMPUTERS>/${null_imputers}/g" -e "s/<EVALUATION_SCENARIO>/${evaluation_scenario}/g" -e "s/<MODEL>/${model}/g" -e "s/<RUN_NUMS>/${run_nums}/g" $TEMPLATE_FILE > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
