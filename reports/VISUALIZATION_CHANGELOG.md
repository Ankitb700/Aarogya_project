# Visualization Changelog

## Files Created

- `reports/executive_summary/executive_scorecard.png`
- `reports/model_comparisons/model_comparison_overview.png`
- `reports/model_comparisons/wpinn_rmse_improvement.png`
- `reports/dashboard_visuals/roi_signal_quality.png`
- `reports/success_cases/video_measurement_examples.png`
- `reports/failure_cases/client_friendly_challenge_gallery.png`
- `reports/presentation_assets/representative_video_frame.png`
- `reports/workflow_visuals/operational_workflow.png`
- `reports/pretrained_analysis/rppg_vbpe_bp_summary.png`
- `reports/pretrained_analysis/rppg_dataset_and_outputs_map.png`
- `reports/pretrained_analysis/wpinn_dataset_story.png`
- `reports/pretrained_analysis/wpinn_business_model_comparison.png`
- `reports/pretrained_analysis/pretrained_analysis_map.png`
- `reports/pretrained_analysis/subject_roi_measurement_overlay.png`
- `reports/pretrained_analysis/roi_technique_panel.png`
- `reports/pretrained_analysis/image_backed_hr_prediction_card.png`
- `reports/pretrained_analysis/image_backed_bp_prediction_card.png`
- `reports/custom_model_results/custom_model_scorecard.png`
- `reports/custom_model_results/custom_label_accuracy_chart.png`
- `reports/custom_model_results/custom_dl_labeled_prediction_card.png`
- `reports/custom_model_results/custom_ml_labeled_prediction_card.png`
- `reports/custom_model_results/custom_wpinn_labeled_result_card.png`
- `reports/dashboard_visuals/client_dashboard.png`
- `reports/presentation_assets/asset_inventory.csv`
- `reports/presentation_assets/ASSET_INVENTORY.md`
- `reports/executive_summary/KEY_FINDINGS.md`

## Files Modified

- `README.md`
- `generate_client_visuals.py`

## Visualizations Added

| Visualization | Purpose | Data source | Output |
|---|---|---|---|
| Executive scorecard | Summarize the project in client-facing KPIs | `outputs/figures/results.json`, prediction CSVs, WPINN CSVs | `reports/executive_summary/executive_scorecard.png` |
| Model comparison overview | Compare held-out HR and BP proxy error | `outputs/predictions/*.csv` | `reports/model_comparisons/model_comparison_overview.png` |
| WPINN improvement chart | Show custom BP-model value versus baseline | `wpin_analysis/WPINN/outputs/model_results.csv` | `reports/model_comparisons/wpinn_rmse_improvement.png` |
| ROI quality chart | Explain which face regions provide clearer measurements | `rppg_inference_vbpe/research_output/roi_summary_table.csv` | `reports/dashboard_visuals/roi_signal_quality.png` |
| Visual measurement examples | Show actual input and tracking outputs | Photos and processed videos in `rppg_inference_vbpe` | `reports/success_cases/video_measurement_examples.png` |
| Challenge gallery | Translate weaker signal cases into plain-language risks | `per_roi_sqi_all.csv` and video frames | `reports/failure_cases/client_friendly_challenge_gallery.png` |
| Operational workflow | Explain the end-to-end process | Existing repository workflow | `reports/workflow_visuals/operational_workflow.png` |
| Client dashboard | One-page visual summary | Generated report assets | `reports/dashboard_visuals/client_dashboard.png` |
| rPPG BP summary | Explain pretrained V-BPE blood-pressure estimates | `rppg_inference_vbpe/SBP_new.csv`, `DBP_new.csv` | `reports/pretrained_analysis/rppg_vbpe_bp_summary.png` |
| rPPG dataset/output map | Show which local assets the pretrained rPPG pipeline uses | `Input_Videos`, `Input_Photos`, `Results/PPG` | `reports/pretrained_analysis/rppg_dataset_and_outputs_map.png` |
| WPINN dataset story | Explain the simulated beat dataset in plain language | `wpin_analysis/WPINN/data/example_data.pkl` | `reports/pretrained_analysis/wpinn_dataset_story.png` |
| WPINN business comparison | Show error reduction from physics-informed modeling | `wpin_analysis/WPINN/outputs/model_results.csv` | `reports/pretrained_analysis/wpinn_business_model_comparison.png` |
| Pretrained analysis map | Compare rPPG, Open-rPPG, and WPINN analysis roles | Project reports and result folders | `reports/pretrained_analysis/pretrained_analysis_map.png` |
| Subject ROI overlay | Mark face, hand, and vessel-path regions on a subject photo | `Input_Photos/SL428.jpg`, rPPG code workflow | `reports/pretrained_analysis/subject_roi_measurement_overlay.png` |
| ROI technique panel | Show original capture, face tracking, and hand tracking examples | Input photo and processed MediaPipe videos | `reports/pretrained_analysis/roi_technique_panel.png` |
| Image-backed HR card | Label actual vs predicted HR beside a project subject image | `outputs/predictions/dl_random_sample_predictions.csv` | `reports/pretrained_analysis/image_backed_hr_prediction_card.png` |
| Image-backed BP card | Label reference vs estimated BP beside the paired subject photo | `Demographic_Data.csv`, `SBP_new.csv`, `DBP_new.csv` | `reports/pretrained_analysis/image_backed_bp_prediction_card.png` |
| Custom model scorecard | Summarize custom ML, DL, and WPINN performance labels | Saved prediction CSVs and WPINN results | `reports/custom_model_results/custom_model_scorecard.png` |
| Custom label accuracy chart | Compare custom model label accuracy and regression errors | `outputs/predictions/*.csv` | `reports/custom_model_results/custom_label_accuracy_chart.png` |
| Custom DL labeled card | Show actual label, predicted label, confidence, and accuracy | `dl_random_sample_predictions.csv` | `reports/custom_model_results/custom_dl_labeled_prediction_card.png` |
| Custom ML labeled card | Show actual label, predicted label, confidence, and accuracy | `ml_random_sample_predictions.csv` | `reports/custom_model_results/custom_ml_labeled_prediction_card.png` |
| Custom WPINN labeled card | Show scenario label, RMSE, and error reduction | `wpin_analysis/WPINN/outputs/model_results.csv` | `reports/custom_model_results/custom_wpinn_labeled_result_card.png` |

## Pending Enhancements

- Add a short slide deck using the generated assets.
- Add more frame-level examples after additional model-output screenshots are available.
- Add confidence-band visuals if future experiments save uncertainty estimates.
