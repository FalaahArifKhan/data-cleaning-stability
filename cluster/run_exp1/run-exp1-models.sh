# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
#     "diabetes hivae exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "diabetes hivae exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "diabetes hivae exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "diabetes hivae mixed_exp rf_clf 1,2,3,4,5,6"
#
#     "german hivae exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "german hivae exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "german hivae exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "german hivae mixed_exp rf_clf 1,2,3,4,5,6"

#     "law_school hivae exp1_mcar3 lr_clf 1,2,3,4,5,6"
#     "law_school hivae exp1_mar3 lr_clf 1,2,3,4,5,6"
#     "law_school hivae exp1_mnar3 lr_clf 1,2,3,4,5,6"
#     "law_school hivae mixed_exp lr_clf 1,2,3,4,5,6"
#
#     "bank hivae exp1_mcar3 lgbm_clf 1,2"
#     "bank hivae exp1_mcar3 lgbm_clf 3,4"
#     "bank hivae exp1_mcar3 lgbm_clf 5,6"
#     "bank hivae exp1_mar3 lgbm_clf 1,2"
#     "bank hivae exp1_mar3 lgbm_clf 3,4"
#     "bank hivae exp1_mar3 lgbm_clf 5,6"
#     "bank hivae exp1_mnar3 lgbm_clf 1,2"
#     "bank hivae exp1_mnar3 lgbm_clf 3,4"
#     "bank hivae exp1_mnar3 lgbm_clf 5,6"
#     "bank hivae mixed_exp lgbm_clf 1,2"
#     "bank hivae mixed_exp lgbm_clf 3,4"
#     "bank hivae mixed_exp lgbm_clf 5,6"

#     "folk hivae exp1_mcar3 mlp_clf 1,2"
#     "folk hivae exp1_mcar3 mlp_clf 3,4"
#     "folk hivae exp1_mcar3 mlp_clf 5,6"
#     "folk hivae exp1_mar3 mlp_clf 1,2"
#     "folk hivae exp1_mar3 mlp_clf 3,4"
#     "folk hivae exp1_mar3 mlp_clf 5,6"
#     "folk hivae exp1_mnar3 mlp_clf 1,2"
#     "folk hivae exp1_mnar3 mlp_clf 3,4"
#     "folk hivae exp1_mnar3 mlp_clf 5,6"
#     "folk hivae mixed_exp mlp_clf 1,2"
#     "folk hivae mixed_exp mlp_clf 3,4"
#     "folk hivae mixed_exp mlp_clf 5,6"

#     "heart hivae exp1_mcar3 gandalf_clf 1,2"
#     "heart hivae exp1_mcar3 gandalf_clf 3,4"
#     "heart hivae exp1_mcar3 gandalf_clf 5,6"
#     "heart hivae exp1_mar3 gandalf_clf 1,2"
#     "heart hivae exp1_mar3 gandalf_clf 3,4"
#     "heart hivae exp1_mar3 gandalf_clf 5,6"
#     "heart hivae exp1_mnar3 gandalf_clf 1,2"
#     "heart hivae exp1_mnar3 gandalf_clf 3,4"
#     "heart hivae exp1_mnar3 gandalf_clf 5,6"
#     "heart hivae mixed_exp gandalf_clf 1,2"
#     "heart hivae mixed_exp gandalf_clf 3,4"
#     "heart hivae mixed_exp gandalf_clf 5,6"



#     "folk_emp hivae exp1_mcar3 gandalf_clf 1"
#     "folk_emp hivae exp1_mcar3 gandalf_clf 2"
#     "folk_emp hivae exp1_mcar3 gandalf_clf 3"
#     "folk_emp hivae exp1_mcar3 gandalf_clf 4"
#     "folk_emp hivae exp1_mcar3 gandalf_clf 5"
#     "folk_emp hivae exp1_mcar3 gandalf_clf 6"
#     "folk_emp hivae exp1_mar3 gandalf_clf 1"
#     "folk_emp hivae exp1_mar3 gandalf_clf 2"
#     "folk_emp hivae exp1_mar3 gandalf_clf 3"
#     "folk_emp hivae exp1_mar3 gandalf_clf 4"
#     "folk_emp hivae exp1_mar3 gandalf_clf 5"
#     "folk_emp hivae exp1_mar3 gandalf_clf 6"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 1"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 2"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 3"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 4"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 5"
#     "folk_emp hivae exp1_mnar3 gandalf_clf 6"
#     "folk_emp hivae mixed_exp gandalf_clf 1"
#     "folk_emp hivae mixed_exp gandalf_clf 2"
#     "folk_emp hivae mixed_exp gandalf_clf 3"
#     "folk_emp hivae mixed_exp gandalf_clf 4"
#     "folk_emp hivae mixed_exp gandalf_clf 5"
#     "folk_emp hivae mixed_exp gandalf_clf 6"
)

TEMPLATE_FILE="../cluster/run_exp1/run-exp1-models-template.sbatch"

# Initialize a counter
index=0

# Iterate through the array of job_configs
for job_config in "${job_configs[@]}"
do
    # Split the job_config into separate variables
    read -r dataset null_imputer evaluation_scenario model run_nums <<< "$job_config"

    # Define the output file name
    output_file="../cluster/run_exp1/sbatch_files/models/run-exp1-${dataset}_${null_imputer}_${evaluation_scenario}_${model}_${index}.sbatch"

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
