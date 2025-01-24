Data Procesing testing (for test_preprocessing):
-

- edge cases handling, rewrite edge cases code and confirm with empty datasets, single user scenarios and unusual rating patterns
- validation of input data and format, test load_ml1m_data functions correctly
- verification od data cleaning operations, ensure that preprocess ratings properly handles missing values
- data transformations, validate rating normalization works correctly across full range of values
- data splitting validation, varify that split_data maintains proper proportions and prevents data leakage

Evaluation Metrics testing:
-

- unit tests for each implementation, test RMSE, MAE, precisionatk, recallatk and ndcg with known input-output pairs
- boundary case validation, verify metric behaviour with perfect conditions, worst-case predictions, and edge cases
- comparason testing, ensure metrics align with expected behaviours (e.g. RMSE should be higher than MAE for the same predictions)
- performance benchmarking, establish baseline metric values and acceptable ranges for model

Model Behaviour testing:
-

- model initalization testing, varify proper setup od embedding layers and weight initialization
- foward pass validation, test prediction generation with controled inputs
- backward pass verification, ensure gradient calculation and updates work correctly

Integration testing:
-

- end-to-end pipeline validation, test full flow from data loading through prediction generation
- API endpoint testing, verify correct handling of recommendation responses