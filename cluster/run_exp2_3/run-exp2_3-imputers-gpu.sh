# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "folk datawig exp2_3_mcar1 1,2"
    "folk datawig exp2_3_mcar1 3,4"
    "folk datawig exp2_3_mcar1 5,6"
    "folk datawig exp2_3_mcar5 1,2"
    "folk datawig exp2_3_mcar5 3,4"
    "folk datawig exp2_3_mcar5 5,6"
    "folk datawig exp2_3_mar1 1,2"
    "folk datawig exp2_3_mar1 3,4"
    "folk datawig exp2_3_mar1 5,6"
    "folk datawig exp2_3_mar5 1,2"
    "folk datawig exp2_3_mar5 3,4"
    "folk datawig exp2_3_mar5 5,6"
    "folk datawig exp2_3_mnar1 1,2"
    "folk datawig exp2_3_mnar1 3,4"
    "folk datawig exp2_3_mnar1 5,6"
    "folk datawig exp2_3_mnar5 1,2"
    "folk datawig exp2_3_mnar5 3,4"
    "folk datawig exp2_3_mnar5 5,6"
    "folk datawig exp2_3_mcar3 1,2"
    "folk datawig exp2_3_mcar3 3,4"
    "folk datawig exp2_3_mcar3 5,6"
    "folk datawig exp2_3_mar3 1,2"
    "folk datawig exp2_3_mar3 3,4"
    "folk datawig exp2_3_mar3 5,6"
    "folk datawig exp2_3_mnar3 1,2"
    "folk datawig exp2_3_mnar3 3,4"
    "folk datawig exp2_3_mnar3 5,6"
)

TEMPLATE_FILE="../cluster/run_exp2_3/run-exp2_3-imputers-gpu-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset null_imputer evaluation_scenario run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_exp2_3/sbatch_files/imputers/run-exp2_3-imputers-${dataset}_${null_imputer}_${evaluation_scenario}_${index}.sbatch"

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
