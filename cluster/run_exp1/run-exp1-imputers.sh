# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "law_school datawig exp1_mcar3 1"
    "law_school datawig exp1_mcar3 2"
    "law_school datawig exp1_mcar3 3"
    "law_school datawig exp1_mcar3 4"
    "law_school datawig exp1_mcar3 5"
    "law_school datawig exp1_mcar3 6"
    "law_school datawig exp1_mar3 1"
    "law_school datawig exp1_mar3 2"
    "law_school datawig exp1_mar3 3"
    "law_school datawig exp1_mar3 4"
    "law_school datawig exp1_mar3 5"
    "law_school datawig exp1_mar3 6"
    "law_school datawig exp1_mnar3 1"
    "law_school datawig exp1_mnar3 2"
    "law_school datawig exp1_mnar3 3"
    "law_school datawig exp1_mnar3 4"
    "law_school datawig exp1_mnar3 5"
    "law_school datawig exp1_mnar3 6"
)

TEMPLATE_FILE="../cluster/run_exp1/run-exp1-imputers-gpu-template.sbatch"

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
