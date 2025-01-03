# Define the list of tuples (dataset_name, model_name, run_nums)
declare -a job_configs=(
#     "diabetes notmiwae exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "diabetes notmiwae exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "diabetes notmiwae exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "diabetes notmiwae mixed_exp rf_clf 1,2,3,4,5,6"
#     "diabetes gain exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "diabetes gain exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "diabetes gain exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "diabetes gain mixed_exp rf_clf 1,2,3,4,5,6"
#     "diabetes tdm exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "diabetes tdm exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "diabetes tdm exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "diabetes tdm mixed_exp rf_clf 1,2,3,4,5,6"
#     "diabetes nomi exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "diabetes nomi exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "diabetes nomi exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "diabetes nomi mixed_exp rf_clf 1,2,3,4,5,6"
#
#     "german notmiwae exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "german notmiwae exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "german notmiwae exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "german notmiwae mixed_exp rf_clf 1,2,3,4,5,6"
#     "german gain exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "german gain exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "german gain exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "german gain mixed_exp rf_clf 1,2,3,4,5,6"
#     "german tdm exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "german tdm exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "german tdm exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "german tdm mixed_exp rf_clf 1,2,3,4,5,6"
#     "german nomi exp1_mcar3 rf_clf 1,2,3,4,5,6"
#     "german nomi exp1_mar3 rf_clf 1,2,3,4,5,6"
#     "german nomi exp1_mnar3 rf_clf 1,2,3,4,5,6"
#     "german nomi mixed_exp rf_clf 1,2,3,4,5,6"

#     "law_school notmiwae exp1_mcar3 lr_clf 1,2,3,4,5,6"
#     "law_school notmiwae exp1_mar3 lr_clf 1,2,3,4,5,6"
#     "law_school notmiwae exp1_mnar3 lr_clf 1,2,3,4,5,6"
#     "law_school notmiwae mixed_exp lr_clf 1,2,3,4,5,6"
#     "law_school gain exp1_mcar3 lr_clf 1,2,3,4,5,6"
#     "law_school gain exp1_mar3 lr_clf 1,2,3,4,5,6"
#     "law_school gain exp1_mnar3 lr_clf 1,2,3,4,5,6"
#     "law_school gain mixed_exp lr_clf 1,2,3,4,5,6"
#     "law_school tdm exp1_mcar3 lr_clf 1,2,3,4,5,6"
#     "law_school tdm exp1_mar3 lr_clf 1,2,3,4,5,6"
#     "law_school tdm exp1_mnar3 lr_clf 1,2,3,4,5,6"
#     "law_school tdm mixed_exp lr_clf 1,2,3,4,5,6"
#     "law_school nomi exp1_mcar3 lr_clf 1,2,3,4,5,6"
#     "law_school nomi exp1_mar3 lr_clf 1,2,3,4,5,6"
#     "law_school nomi exp1_mnar3 lr_clf 1,2,3,4,5,6"
#     "law_school nomi mixed_exp lr_clf 1,2,3,4,5,6"
#
#     "bank notmiwae exp1_mcar3 lgbm_clf 1,2"
#     "bank notmiwae exp1_mcar3 lgbm_clf 3,4"
#     "bank notmiwae exp1_mcar3 lgbm_clf 5,6"
#     "bank notmiwae exp1_mar3 lgbm_clf 1,2"
#     "bank notmiwae exp1_mar3 lgbm_clf 3,4"
#     "bank notmiwae exp1_mar3 lgbm_clf 5,6"
#     "bank notmiwae exp1_mnar3 lgbm_clf 1,2"
#     "bank notmiwae exp1_mnar3 lgbm_clf 3,4"
#     "bank notmiwae exp1_mnar3 lgbm_clf 5,6"
#     "bank notmiwae mixed_exp lgbm_clf 1,2"
#     "bank notmiwae mixed_exp lgbm_clf 3,4"
#     "bank notmiwae mixed_exp lgbm_clf 5,6"
#     "bank gain exp1_mcar3 lgbm_clf 1,2"
#     "bank gain exp1_mcar3 lgbm_clf 3,4"
#     "bank gain exp1_mcar3 lgbm_clf 5,6"
#     "bank gain exp1_mar3 lgbm_clf 1,2"
#     "bank gain exp1_mar3 lgbm_clf 3,4"
#     "bank gain exp1_mar3 lgbm_clf 5,6"
#     "bank gain exp1_mnar3 lgbm_clf 1,2"
#     "bank gain exp1_mnar3 lgbm_clf 3,4"
#     "bank gain exp1_mnar3 lgbm_clf 5,6"
#     "bank gain mixed_exp lgbm_clf 1,2"
#     "bank gain mixed_exp lgbm_clf 3,4"
#     "bank gain mixed_exp lgbm_clf 5,6"
#     "bank tdm exp1_mcar3 lgbm_clf 1,2"
#     "bank tdm exp1_mcar3 lgbm_clf 3,4"
#     "bank tdm exp1_mcar3 lgbm_clf 5,6"
#     "bank tdm exp1_mar3 lgbm_clf 1,2"
#     "bank tdm exp1_mar3 lgbm_clf 3,4"
#     "bank tdm exp1_mar3 lgbm_clf 5,6"
#     "bank tdm exp1_mnar3 lgbm_clf 1,2"
#     "bank tdm exp1_mnar3 lgbm_clf 3,4"
#     "bank tdm exp1_mnar3 lgbm_clf 5,6"
#     "bank tdm mixed_exp lgbm_clf 1,2"
#     "bank tdm mixed_exp lgbm_clf 3,4"
#     "bank tdm mixed_exp lgbm_clf 5,6"
#     "bank nomi exp1_mcar3 lgbm_clf 1,2"
#     "bank nomi exp1_mcar3 lgbm_clf 3,4"
#     "bank nomi exp1_mcar3 lgbm_clf 5,6"
#     "bank nomi exp1_mar3 lgbm_clf 1,2"
#     "bank nomi exp1_mar3 lgbm_clf 3,4"
#     "bank nomi exp1_mar3 lgbm_clf 5,6"
#     "bank nomi exp1_mnar3 lgbm_clf 1,2"
#     "bank nomi exp1_mnar3 lgbm_clf 3,4"
#     "bank nomi exp1_mnar3 lgbm_clf 5,6"
#     "bank nomi mixed_exp lgbm_clf 1,2"
#     "bank nomi mixed_exp lgbm_clf 3,4"
#     "bank nomi mixed_exp lgbm_clf 5,6"

#     "folk notmiwae exp1_mcar3 mlp_clf 1,2"
#     "folk notmiwae exp1_mcar3 mlp_clf 3,4"
#     "folk notmiwae exp1_mcar3 mlp_clf 5,6"
#     "folk notmiwae exp1_mar3 mlp_clf 1,2"
#     "folk notmiwae exp1_mar3 mlp_clf 3,4"
#     "folk notmiwae exp1_mar3 mlp_clf 5,6"
#     "folk notmiwae exp1_mnar3 mlp_clf 1,2"
#     "folk notmiwae exp1_mnar3 mlp_clf 3,4"
#     "folk notmiwae exp1_mnar3 mlp_clf 5,6"
#     "folk notmiwae mixed_exp mlp_clf 1,2"
#     "folk notmiwae mixed_exp mlp_clf 3,4"
#     "folk notmiwae mixed_exp mlp_clf 5,6"
#     "folk gain exp1_mcar3 mlp_clf 1,2"
#     "folk gain exp1_mcar3 mlp_clf 3,4"
#     "folk gain exp1_mcar3 mlp_clf 5,6"
#     "folk gain exp1_mar3 mlp_clf 1,2"
#     "folk gain exp1_mar3 mlp_clf 3,4"
#     "folk gain exp1_mar3 mlp_clf 5,6"
#     "folk gain exp1_mnar3 mlp_clf 1,2"
#     "folk gain exp1_mnar3 mlp_clf 3,4"
#     "folk gain exp1_mnar3 mlp_clf 5,6"
#     "folk gain mixed_exp mlp_clf 1,2"
#     "folk gain mixed_exp mlp_clf 3,4"
#     "folk gain mixed_exp mlp_clf 5,6"
#     "folk tdm exp1_mcar3 mlp_clf 1,2"
#     "folk tdm exp1_mcar3 mlp_clf 3,4"
#     "folk tdm exp1_mcar3 mlp_clf 5,6"
#     "folk tdm exp1_mar3 mlp_clf 1,2"
#     "folk tdm exp1_mar3 mlp_clf 3,4"
#     "folk tdm exp1_mar3 mlp_clf 5,6"
#     "folk tdm exp1_mnar3 mlp_clf 1,2"
#     "folk tdm exp1_mnar3 mlp_clf 3,4"
#     "folk tdm exp1_mnar3 mlp_clf 5,6"
#     "folk tdm mixed_exp mlp_clf 1,2"
#     "folk tdm mixed_exp mlp_clf 3,4"
#     "folk tdm mixed_exp mlp_clf 5,6"
#     "folk nomi exp1_mcar3 mlp_clf 1,2"
#     "folk nomi exp1_mcar3 mlp_clf 3,4"
#     "folk nomi exp1_mcar3 mlp_clf 5,6"
#     "folk nomi exp1_mar3 mlp_clf 1,2"
#     "folk nomi exp1_mar3 mlp_clf 3,4"
#     "folk nomi exp1_mar3 mlp_clf 5,6"
#     "folk nomi exp1_mnar3 mlp_clf 1,2"
#     "folk nomi exp1_mnar3 mlp_clf 3,4"
#     "folk nomi exp1_mnar3 mlp_clf 5,6"
#     "folk nomi mixed_exp mlp_clf 1,2"
#     "folk nomi mixed_exp mlp_clf 3,4"
#     "folk nomi mixed_exp mlp_clf 5,6"

#     "heart notmiwae exp1_mcar3 gandalf_clf 1,2"
#     "heart notmiwae exp1_mcar3 gandalf_clf 3,4"
#     "heart notmiwae exp1_mcar3 gandalf_clf 5,6"
#     "heart notmiwae exp1_mar3 gandalf_clf 1,2"
#     "heart notmiwae exp1_mar3 gandalf_clf 3,4"
#     "heart notmiwae exp1_mar3 gandalf_clf 5,6"
#     "heart notmiwae exp1_mnar3 gandalf_clf 1,2"
#     "heart notmiwae exp1_mnar3 gandalf_clf 3,4"
#     "heart notmiwae exp1_mnar3 gandalf_clf 5,6"
#     "heart notmiwae mixed_exp gandalf_clf 1,2"
#     "heart notmiwae mixed_exp gandalf_clf 3,4"
#     "heart notmiwae mixed_exp gandalf_clf 5,6"
#     "heart gain exp1_mcar3 gandalf_clf 1,2"
#     "heart gain exp1_mcar3 gandalf_clf 3,4"
#     "heart gain exp1_mcar3 gandalf_clf 5,6"
#     "heart gain exp1_mar3 gandalf_clf 1,2"
#     "heart gain exp1_mar3 gandalf_clf 3,4"
#     "heart gain exp1_mar3 gandalf_clf 5,6"
#     "heart gain exp1_mnar3 gandalf_clf 1,2"
#     "heart gain exp1_mnar3 gandalf_clf 3,4"
#     "heart gain exp1_mnar3 gandalf_clf 5,6"
#     "heart gain mixed_exp gandalf_clf 1,2"
#     "heart gain mixed_exp gandalf_clf 3,4"
#     "heart gain mixed_exp gandalf_clf 5,6"
#     "heart tdm exp1_mcar3 gandalf_clf 1,2"
#     "heart tdm exp1_mcar3 gandalf_clf 3,4"
#     "heart tdm exp1_mcar3 gandalf_clf 5,6"
#     "heart tdm exp1_mar3 gandalf_clf 1,2"
#     "heart tdm exp1_mar3 gandalf_clf 3,4"
#     "heart tdm exp1_mar3 gandalf_clf 5,6"
#     "heart tdm exp1_mnar3 gandalf_clf 1,2"
#     "heart tdm exp1_mnar3 gandalf_clf 3,4"
#     "heart tdm exp1_mnar3 gandalf_clf 5,6"
#     "heart tdm mixed_exp gandalf_clf 1,2"
#     "heart tdm mixed_exp gandalf_clf 3,4"
#     "heart tdm mixed_exp gandalf_clf 5,6"
#     "heart nomi exp1_mcar3 gandalf_clf 1,2"
#     "heart nomi exp1_mcar3 gandalf_clf 3,4"
#     "heart nomi exp1_mcar3 gandalf_clf 5,6"
#     "heart nomi exp1_mar3 gandalf_clf 1,2"
#     "heart nomi exp1_mar3 gandalf_clf 3,4"
#     "heart nomi exp1_mar3 gandalf_clf 5,6"
#     "heart nomi exp1_mnar3 gandalf_clf 1,2"
#     "heart nomi exp1_mnar3 gandalf_clf 3,4"
#     "heart nomi exp1_mnar3 gandalf_clf 5,6"
#     "heart nomi mixed_exp gandalf_clf 1,2"
#     "heart nomi mixed_exp gandalf_clf 3,4"
#     "heart nomi mixed_exp gandalf_clf 5,6"
#
#
#
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 1"
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 2"
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 3"
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 4"
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 5"
#     "folk_emp notmiwae exp1_mcar3 gandalf_clf 6"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 1"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 2"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 3"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 4"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 5"
#     "folk_emp notmiwae exp1_mar3 gandalf_clf 6"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 1"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 2"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 3"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 4"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 5"
#     "folk_emp notmiwae exp1_mnar3 gandalf_clf 6"
#     "folk_emp notmiwae mixed_exp gandalf_clf 1"
#     "folk_emp notmiwae mixed_exp gandalf_clf 2"
#     "folk_emp notmiwae mixed_exp gandalf_clf 3"
#     "folk_emp notmiwae mixed_exp gandalf_clf 4"
#     "folk_emp notmiwae mixed_exp gandalf_clf 5"
#     "folk_emp notmiwae mixed_exp gandalf_clf 6"
#
#     "folk_emp gain exp1_mcar3 gandalf_clf 1"
#     "folk_emp gain exp1_mcar3 gandalf_clf 2"
#     "folk_emp gain exp1_mcar3 gandalf_clf 3"
#     "folk_emp gain exp1_mcar3 gandalf_clf 4"
#     "folk_emp gain exp1_mcar3 gandalf_clf 5"
#     "folk_emp gain exp1_mcar3 gandalf_clf 6"
#     "folk_emp gain exp1_mar3 gandalf_clf 1"
#     "folk_emp gain exp1_mar3 gandalf_clf 2"
#     "folk_emp gain exp1_mar3 gandalf_clf 3"
#     "folk_emp gain exp1_mar3 gandalf_clf 4"
#     "folk_emp gain exp1_mar3 gandalf_clf 5"
#     "folk_emp gain exp1_mar3 gandalf_clf 6"
#     "folk_emp gain exp1_mnar3 gandalf_clf 1"
#     "folk_emp gain exp1_mnar3 gandalf_clf 2"
#     "folk_emp gain exp1_mnar3 gandalf_clf 3"
#     "folk_emp gain exp1_mnar3 gandalf_clf 4"
#     "folk_emp gain exp1_mnar3 gandalf_clf 5"
#     "folk_emp gain exp1_mnar3 gandalf_clf 6"
#     "folk_emp gain mixed_exp gandalf_clf 1"
#     "folk_emp gain mixed_exp gandalf_clf 2"
#     "folk_emp gain mixed_exp gandalf_clf 3"
#     "folk_emp gain mixed_exp gandalf_clf 4"
#     "folk_emp gain mixed_exp gandalf_clf 5"
#     "folk_emp gain mixed_exp gandalf_clf 6"
#
#     "folk_emp tdm exp1_mcar3 gandalf_clf 1"
#     "folk_emp tdm exp1_mcar3 gandalf_clf 2"
#     "folk_emp tdm exp1_mcar3 gandalf_clf 3"
#     "folk_emp tdm exp1_mcar3 gandalf_clf 4"
#     "folk_emp tdm exp1_mcar3 gandalf_clf 5"
#     "folk_emp tdm exp1_mcar3 gandalf_clf 6"
#     "folk_emp tdm exp1_mar3 gandalf_clf 1"
#     "folk_emp tdm exp1_mar3 gandalf_clf 2"
#     "folk_emp tdm exp1_mar3 gandalf_clf 3"
#     "folk_emp tdm exp1_mar3 gandalf_clf 4"
#     "folk_emp tdm exp1_mar3 gandalf_clf 5"
#     "folk_emp tdm exp1_mar3 gandalf_clf 6"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 1"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 2"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 3"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 4"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 5"
#     "folk_emp tdm exp1_mnar3 gandalf_clf 6"
#     "folk_emp tdm mixed_exp gandalf_clf 1"
#     "folk_emp tdm mixed_exp gandalf_clf 2"
#     "folk_emp tdm mixed_exp gandalf_clf 3"
#     "folk_emp tdm mixed_exp gandalf_clf 4"
#     "folk_emp tdm mixed_exp gandalf_clf 5"
#     "folk_emp tdm mixed_exp gandalf_clf 6"

     "folk_emp nomi exp1_mcar3 gandalf_clf 1"
     "folk_emp nomi exp1_mcar3 gandalf_clf 2"
     "folk_emp nomi exp1_mcar3 gandalf_clf 3"
     "folk_emp nomi exp1_mcar3 gandalf_clf 4"
     "folk_emp nomi exp1_mcar3 gandalf_clf 5"
     "folk_emp nomi exp1_mcar3 gandalf_clf 6"
     "folk_emp nomi exp1_mar3 gandalf_clf 1"
     "folk_emp nomi exp1_mar3 gandalf_clf 2"
     "folk_emp nomi exp1_mar3 gandalf_clf 3"
     "folk_emp nomi exp1_mar3 gandalf_clf 4"
     "folk_emp nomi exp1_mar3 gandalf_clf 5"
     "folk_emp nomi exp1_mar3 gandalf_clf 6"
     "folk_emp nomi exp1_mnar3 gandalf_clf 1"
     "folk_emp nomi exp1_mnar3 gandalf_clf 2"
     "folk_emp nomi exp1_mnar3 gandalf_clf 3"
     "folk_emp nomi exp1_mnar3 gandalf_clf 4"
     "folk_emp nomi exp1_mnar3 gandalf_clf 5"
     "folk_emp nomi exp1_mnar3 gandalf_clf 6"
     "folk_emp nomi mixed_exp gandalf_clf 1"
     "folk_emp nomi mixed_exp gandalf_clf 2"
     "folk_emp nomi mixed_exp gandalf_clf 3"
     "folk_emp nomi mixed_exp gandalf_clf 4"
     "folk_emp nomi mixed_exp gandalf_clf 5"
     "folk_emp nomi mixed_exp gandalf_clf 6"
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
