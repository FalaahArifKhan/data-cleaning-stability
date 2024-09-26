// ==============================================================
// Validate plots for imputation performance for all datasets
// ==============================================================
// For F1 score and RMSE
db.imputation_performance_metrics.aggregate([
  // Match stage to filter based on the conditions
  {
    $match: {
      dataset_name: 'german',
      evaluation_scenario: 'mixed_exp',
      tag: 'OK',
      subgroup: 'overall',
      column_type: 'numerical',
//      column_type: 'categorical',
      dataset_part: 'X_test_MCAR1 & MAR1 & MNAR1'
    }
  },
  // Group by the specified fields and calculate the average f1_score
  {
    $group: {
      _id: {
        dataset_name: "$dataset_name",
        evaluation_scenario: "$evaluation_scenario",
        column_type: "$column_type",
        dataset_part: "$dataset_part",
        null_imputer_name: "$null_imputer_name",
        experiment_seed: "$experiment_seed"
      },
      avg_rmse: { $avg: "$rmse" }
    }
  },
  // Sort by the specified fields
  {
    $sort: {
      "_id.dataset_name": 1,
      "_id.evaluation_scenario": 1,
      "_id.column_type": 1,
      "_id.dataset_part": 1,
      "_id.null_imputer_name": 1,
      "_id.experiment_seed": 1
    }
  },
  // Optionally, you can project the fields to reshape the output (optional)
  {
    $project: {
      _id: 0,
      dataset_name: "$_id.dataset_name",
      evaluation_scenario: "$_id.evaluation_scenario",
      column_type: "$_id.column_type",
      dataset_part: "$_id.dataset_part",
      null_imputer_name: "$_id.null_imputer_name",
      experiment_seed: "$_id.experiment_seed",
      avg_rmse: 1
    }
  }
]);


// KL Divergence
db.imputation_performance_metrics.aggregate([
  // Match stage to filter based on the conditions
  {
    $match: {
      dataset_name: 'german',
      evaluation_scenario: 'mixed_exp',
      tag: 'OK',
      subgroup: 'overall',
//      column_type: 'categorical',
      dataset_part: 'X_test_MCAR1 & MAR1 & MNAR1'
    }
  },
  // Group by the specified fields and calculate the average f1_score
  {
    $group: {
      _id: {
        dataset_name: "$dataset_name",
        evaluation_scenario: "$evaluation_scenario",
//        column_type: "$column_type",
        dataset_part: "$dataset_part",
        null_imputer_name: "$null_imputer_name",
        experiment_seed: "$experiment_seed"
      },
      avg_kl_divergence_pred: { $avg: "$kl_divergence_pred" }
    }
  },
  // Sort by the specified fields
  {
    $sort: {
      "_id.dataset_name": 1,
      "_id.evaluation_scenario": 1,
//      "_id.column_type": 1,
      "_id.dataset_part": 1,
      "_id.null_imputer_name": 1,
      "_id.experiment_seed": 1
    }
  },
  // Optionally, you can project the fields to reshape the output (optional)
  {
    $project: {
      _id: 0,
      dataset_name: "$_id.dataset_name",
      evaluation_scenario: "$_id.evaluation_scenario",
//      column_type: "$_id.column_type",
      dataset_part: "$_id.dataset_part",
      null_imputer_name: "$_id.null_imputer_name",
      experiment_seed: "$_id.experiment_seed",
      avg_kl_divergence_pred: 1
    }
  }
]);
