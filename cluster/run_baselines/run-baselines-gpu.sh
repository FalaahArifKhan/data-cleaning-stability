# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "heart gandalf_clf 1,2"
    "heart gandalf_clf 3,4"
    "heart gandalf_clf 5,6"
)

TEMPLATE_FILE="../cluster/run_baselines/run-baselines-gpu-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset model run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_baselines/sbatch_files/run-baselines-${dataset}_${model}_${index}.sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<DATASET>/${dataset}/g" -e "s/<MODEL>/${model}/g" -e "s/<RUN_NUMS>/${run_nums}/g" $TEMPLATE_FILE > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done