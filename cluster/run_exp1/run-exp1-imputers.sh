# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
    "german gain exp1_mcar3 1,2,3"
    "german gain exp1_mcar3 4,5,6"
    "german gain exp1_mar3 1,2,3"
    "german gain exp1_mar3 4,5,6"
    "german gain exp1_mnar3 1,2,3"
    "german gain exp1_mnar3 4,5,6"
    "german gain mixed_exp 1,2,3"
    "german gain mixed_exp 4,5,6"

    "german tdm exp1_mcar3 1,2,3"
    "german tdm exp1_mcar3 4,5,6"
    "german tdm exp1_mar3 1,2,3"
    "german tdm exp1_mar3 4,5,6"
    "german tdm exp1_mnar3 1,2,3"
    "german tdm exp1_mnar3 4,5,6"
    "german tdm mixed_exp 1,2,3"
    "german tdm mixed_exp 4,5,6"

    "law_school gain exp1_mcar3 1,2"
    "law_school gain exp1_mcar3 3,4"
    "law_school gain exp1_mcar3 5,6"
    "law_school gain exp1_mar3 1,2"
    "law_school gain exp1_mar3 3,4"
    "law_school gain exp1_mar3 5,6"
    "law_school gain exp1_mnar3 1,2"
    "law_school gain exp1_mnar3 3,4"
    "law_school gain exp1_mnar3 5,6"
    "law_school gain mixed_exp 1,2"
    "law_school gain mixed_exp 3,4"
    "law_school gain mixed_exp 5,6"

    "law_school tdm exp1_mcar3 1,2"
    "law_school tdm exp1_mcar3 3,4"
    "law_school tdm exp1_mcar3 5,6"
    "law_school tdm exp1_mar3 1,2"
    "law_school tdm exp1_mar3 3,4"
    "law_school tdm exp1_mar3 5,6"
    "law_school tdm exp1_mnar3 1,2"
    "law_school tdm exp1_mnar3 3,4"
    "law_school tdm exp1_mnar3 5,6"
    "law_school tdm mixed_exp 1,2"
    "law_school tdm mixed_exp 3,4"
    "law_school tdm mixed_exp 5,6"

    "folk gain exp1_mcar3 1,2"
    "folk gain exp1_mcar3 3,4"
    "folk gain exp1_mcar3 5,6"
    "folk gain exp1_mar3 1,2"
    "folk gain exp1_mar3 3,4"
    "folk gain exp1_mar3 5,6"
    "folk gain exp1_mnar3 1,2"
    "folk gain exp1_mnar3 3,4"
    "folk gain exp1_mnar3 5,6"
    "folk gain mixed_exp 1,2"
    "folk gain mixed_exp 3,4"
    "folk gain mixed_exp 5,6"

    "folk tdm exp1_mcar3 1"
    "folk tdm exp1_mcar3 2"
    "folk tdm exp1_mcar3 3"
    "folk tdm exp1_mcar3 4"
    "folk tdm exp1_mcar3 5"
    "folk tdm exp1_mcar3 6"
    "folk tdm exp1_mar3 1"
    "folk tdm exp1_mar3 2"
    "folk tdm exp1_mar3 3"
    "folk tdm exp1_mar3 4"
    "folk tdm exp1_mar3 5"
    "folk tdm exp1_mar3 6"
    "folk tdm exp1_mnar3 1"
    "folk tdm exp1_mnar3 2"
    "folk tdm exp1_mnar3 3"
    "folk tdm exp1_mnar3 4"
    "folk tdm exp1_mnar3 5"
    "folk tdm exp1_mnar3 6"
    "folk tdm mixed_exp 1"
    "folk tdm mixed_exp 2"
    "folk tdm mixed_exp 3"
    "folk tdm mixed_exp 4"
    "folk tdm mixed_exp 5"
    "folk tdm mixed_exp 6"

#    "bank gain exp1_mcar3 1,2,3"
#    "bank gain exp1_mcar3 4,5,6"
#    "bank gain exp1_mar3 1,2,3"
#    "bank gain exp1_mar3 4,5,6"
#    "bank gain exp1_mnar3 1,2,3"
#    "bank gain exp1_mnar3 4,5,6"
#    "bank gain mixed_exp 1,2,3"
#    "bank gain mixed_exp 4,5,6"
#
#    "bank tdm exp1_mcar3 1,2,3"
#    "bank tdm exp1_mcar3 4,5,6"
#    "bank tdm exp1_mar3 1,2,3"
#    "bank tdm exp1_mar3 4,5,6"
#    "bank tdm exp1_mnar3 1,2,3"
#    "bank tdm exp1_mnar3 4,5,6"
#    "bank tdm mixed_exp 1,2,3"
#    "bank tdm mixed_exp 4,5,6"
#
#    "heart gain exp1_mcar3 1,2,3"
#    "heart gain exp1_mcar3 4,5,6"
#    "heart gain exp1_mar3 1,2,3"
#    "heart gain exp1_mar3 4,5,6"
#    "heart gain exp1_mnar3 1,2,3"
#    "heart gain exp1_mnar3 4,5,6"
#    "heart gain mixed_exp 1,2,3"
#    "heart gain mixed_exp 4,5,6"
#
#    "heart tdm exp1_mcar3 1,2,3"
#    "heart tdm exp1_mcar3 4,5,6"
#    "heart tdm exp1_mar3 1,2,3"
#    "heart tdm exp1_mar3 4,5,6"
#    "heart tdm exp1_mnar3 1,2,3"
#    "heart tdm exp1_mnar3 4,5,6"
#    "heart tdm mixed_exp 1,2,3"
#    "heart tdm mixed_exp 4,5,6"
#
#    "folk_emp gain exp1_mcar3 1,2,3"
#    "folk_emp gain exp1_mcar3 4,5,6"
#    "folk_emp gain exp1_mar3 1,2,3"
#    "folk_emp gain exp1_mar3 4,5,6"
#    "folk_emp gain exp1_mnar3 1,2,3"
#    "folk_emp gain exp1_mnar3 4,5,6"
#    "folk_emp gain mixed_exp 1,2,3"
#    "folk_emp gain mixed_exp 4,5,6"
#
#    "folk_emp tdm exp1_mcar3 1,2,3"
#    "folk_emp tdm exp1_mcar3 4,5,6"
#    "folk_emp tdm exp1_mar3 1,2,3"
#    "folk_emp tdm exp1_mar3 4,5,6"
#    "folk_emp tdm exp1_mnar3 1,2,3"
#    "folk_emp tdm exp1_mnar3 4,5,6"
#    "folk_emp tdm mixed_exp 1,2,3"
#    "folk_emp tdm mixed_exp 4,5,6"
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
