# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "diabetes miss_forest exp2_3_mcar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mcar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mcar3 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar3 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar3 1,2,3,4,5,6"

    "diabetes miss_forest exp2_3_mcar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mcar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar1 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar5 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mcar3 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mar3 1,2,3,4,5,6"
    "diabetes miss_forest exp2_3_mnar3 1,2,3,4,5,6"
)

TEMPLATE_FILE="../cluster/run_exp1/run-exp1-imputers-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset null_imputer evaluation_scenario run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_exp1/sbatch_files/imputers/run-exp1-imputers-${dataset}_${null_imputer}_${evaluation_scenario}_${index}.sbatch"

    # Create an empty file
    touch $output_file

    # Use sed to replace placeholders with actual values
    sed -e "s/<DATASET>/${dataset}/g" -e "s/<EVALUATION_SCENARIO>/${evaluation_scenario}/g" -e "s/<NULL_IMPUTER>/${null_imputer}/g" -e "s/<RUN_NUMS>/${run_nums}/g" $TEMPLATE_FILE > $output_file

    # Execute a SLURM job
    sbatch $output_file

    echo "Job was executed: $output_file"

    # Increment the index
    ((index++))
done
