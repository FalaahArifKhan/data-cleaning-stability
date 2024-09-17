// ==============================================================
// Find all successfully executed experiments for specific model
// ==============================================================
db.exp_nulls_data_cleaning.aggregate([
  {
    $match: {
      model_name: 'gandalf_clf',
      tag: 'OK'
    }
  },
  {
    $group: {
      _id: {
        dataset_name: "$dataset_name",
        null_imputer_name: "$null_imputer_name",
        evaluation_scenario: "$evaluation_scenario",
        model_name: "$model_name",
        experiment_iteration: "$experiment_iteration"
      }
    }
  },
  {
    $sort: {
      "_id.dataset_name": 1,
      "_id.null_imputer_name": 1,
      "_id.evaluation_scenario": 1,
      "_id.experiment_iteration": 1
    }
  },
  {
    $project: {
      _id: 0,
      dataset_name: "$_id.dataset_name",
      null_imputer_name: "$_id.null_imputer_name",
      evaluation_scenario: "$_id.evaluation_scenario",
      model_name: "$_id.model_name",
      experiment_iteration: "$_id.experiment_iteration"
    }
  }
]);


// ==========================================================================================================
// Find best models based on F1 for specific dataset and each experiment iteration.
// If model results are the same, two models will be displayed per one experiment iteration as the best ones.
// ==========================================================================================================
db.exp_nulls_data_cleaning.aggregate([
  {
    $match: { dataset_name: "law_school", evaluation_scenario: "baseline", model_name: {$ne: "tabpfn_clf"}, metric: "F1", subgroup: "overall", tag: "OK" }
  },
  {
    $group: {
      _id: "$experiment_iteration",                 // Group by 'group'
      maxTotalValue: { $max: "$metric_value" }      // Find the maximum total_value for each group
    }
  },
  {
    $lookup: {                                     // Use $lookup to join original documents back to max values
      from: "exp_nulls_data_cleaning",             // Perform a self-join
      let: { groupId: "$_id", maxValue: "$maxTotalValue" },
      pipeline: [
        { $match: {
            $expr: {
              $and: [
                { $eq: ["$experiment_iteration", "$$groupId"] },     // Match group
                { $eq: ["$metric_value", "$$maxValue"] },            // Match max value
                { $eq: ["$dataset_name", "law_school"] },
                { $eq: ["$evaluation_scenario", "baseline"] },
                { $eq: ["$metric", "F1"] },
                { $eq: ["$subgroup", "overall"] },
                { $eq: ["$tag", "OK"] },
                { $ne: ["$model_name", "tabpfn_clf"] }
              ]
            }
        }}
      ],
      as: "maxValueDocs"
    }
  },
  {
    $unwind: "$maxValueDocs"                        // Unwind the array to return documents
  },
  {
    $replaceRoot: { newRoot: "$maxValueDocs" }      // Replace root to return original documents
  },
  {
    $project: { _id: 0, dataset_split_seed: 0, exp_pipeline_guid: 0, model_params: 0, model_init_seed: 0, null_imputer_name: 0 }   // Optionally, hide the total_value field from output
  }
]);
