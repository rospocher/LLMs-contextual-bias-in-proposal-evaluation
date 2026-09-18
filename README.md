# Online Materials

Online materials for the manuscript *Contextual Cue Sensitivity in LLM-Based Proposal Scoring: A Controlled ERC-Style Study*.
This repository contains the data, scripts, and analysis outputs supporting the study.

This work has been conducted within the [Digital Arena for Inclusive Humanities](https://daih.eu) of the University of Verona, Italy.

Full reference to the paper:
```bibtex
@inproceedings{2026aacl,
	author = {Marco Rospocher},
	booktitle = {Proceedings of the The 5th Asia-Pacific Chapter of the Association for Computational Linguistics & the 15th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2026)},
	title = {Contextual Cue Sensitivity in LLM Scoring of Project Proposals: A Controlled ERC-Style Study},
	year = {To appear},
}
```


## Repository structure

```text
.
├── analysis_output/
├── code/
│   └── dataset_creation/
└── generated_answers/
```

## Contents

### `generated_answers/`
Raw model-generated scoring outputs used as inputs to the analyses.

- `allScores_allModels_prompt01.csv`
- `allScores_allModels_prompt02.csv`
- `allScores_allModels_prompt03.csv`

Each file corresponds to one prompt template and contains the scored responses produced by all the evaluated models.

### `code/`
Scripts used to generate model answers, preprocess and merge outputs, run the statistical analyses, and produce the figures and summary tables.

#### `code/dataset_creation/`
Scripts used to construct the synthetic ERC-style proposal texts employed as experimental stimuli. The folder contains also the file `doi-executive-summaries.txt` with the DOI of the executive summaries used for the study.

### `analysis_output/`
Processed analysis outputs, including pooled summaries, model- and prompt-specific breakdowns, significance tests, robustness summaries, and figures.

#### Main pooled outputs
- `baseline_dumbbell_questions_pooled.png`: pooled dumbbell plot of score shifts by question relative to the blind baseline
- `coherence_q06_prompt_vs_model.png`: prompt-vs-model coherence plot for the overall score (`q06`)
- `mixed_effects_q06.csv`: mixed-effects estimates for blind-referenced effects on `q06`
- `mixed_effects_q06_pair_differences.csv`: mixed-effects estimates for paired contrast differences on `q06`
- `pooled_question_effects.csv`: pooled question-level effect estimates across models and prompts
- `text_level_effects.csv`: proposal-level effect estimates underlying the aggregate summaries

#### Criterion-level and q06 consistency diagnostics
- `question_group_effects_q01q04_vs_q05q06.csv`: grouped contextual effects separating project-focused criteria (`q01`-`q04`) from capacity/global criteria (`q05`-`q06`)
- `q06_consistency_with_q01_q05.csv`: correlations between `q06` and the mean of `q01`-`q05`, both for raw scores and blind-referenced effects

#### Model-level summaries
- `model_level_effects_compact.csv`: compact effect estimates by model
- `model_pair_difference_effects.csv`: paired contrast differences by model
- `model_robustness_summary.csv`: robustness of main effects across models
- `model_pair_robustness_summary.csv`: robustness of paired contrast differences across models

#### Prompt-level summaries
- `prompt_level_effects_compact.csv`: compact effect estimates by prompt template
- `prompt_pair_difference_effects.csv`: paired contrast differences by prompt template
- `prompt_robustness_summary.csv`: robustness of main effects across prompts
- `prompt_pair_robustness_summary.csv`: robustness of paired contrast differences across prompts

#### Pairwise significance analyses
- `pairwise_significance.csv`: paired statistical comparisons between contrast effects
- `pairwise_significance_permutation.csv`: permutation-based comparisons between contrast effects

#### PI-cue interaction analysis
- `pi_interactions_q06.csv`: two-way interaction models for `q06` in the 2x2x2 PI-cue design

#### Repeated-run variability
- `within_cell_run_variability.csv`: within-cell variability across the five repeated valid runs at temperature 0.0. Cells are defined by model, proposal, prompt template, condition, and question.

#### Proposal-level q06 robustness
- `proposal_level_q06_sign_consistency.csv`: proposal-level sign consistency for `q06` paired contrasts after averaging over models and prompt templates
- `proposal_level_q06_bootstrap.csv`: proposal-level bootstrap intervals for `q06` paired contrasts
- `proposal_level_q06_leave_one_out.csv`: leave-one-proposal-out robustness checks for `q06` paired contrasts

#### Model-specific figures
`per_model_question_dumbbell_plots/` contains question-level dumbbell plots computed separately for each model:

- `baseline_dumbbell_questions_deepseek.png`
- `baseline_dumbbell_questions_gemma.png`
- `baseline_dumbbell_questions_glm.png`
- `baseline_dumbbell_questions_gpt.png`
- `baseline_dumbbell_questions_kimi.png`
- `baseline_dumbbell_questions_ministral.png`
- `baseline_dumbbell_questions_mistral.png`
- `baseline_dumbbell_questions_phi.png`

#### Prompt-specific figures
`per_prompt_question_dumbbell_plots/` contains question-level dumbbell plots computed separately for each prompt template:

- `baseline_dumbbell_questions_prompt_A.png`
- `baseline_dumbbell_questions_prompt_B.png`
- `baseline_dumbbell_questions_prompt_C.png`

#### Rank-selection analysis
`rank_selection/` contains outputs examining how cue-induced score shifts affect proposal rankings and top-`K` selection.

- `aggregated_scores_per_condition.csv`: aggregated proposal scores by condition
- `bootstrap_exit_rate_ci_by_effect_and_k.csv`: bootstrap confidence intervals for exit rates from the funded set
- `bootstrap_rank_loss_ci_by_effect_and_rank.csv`: bootstrap confidence intervals for rank loss by baseline rank
- `mean_rank_loss_by_base_rank.png`: mean rank-change profile by baseline rank
- `model_consistency_summary.csv`: consistency of rank-selection results across models
- `prompt_consistency_summary.csv`: consistency of rank-selection results across prompts
- `selection_summary_two_panel.png`: main summary figure for the rank-selection analysis
- `summary_by_effect_and_k.csv`: pooled summary by effect and selection threshold `K`
- `summary_by_effect_k_and_model.csv`: model-specific top-`K` summaries
- `summary_by_effect_k_and_prompt.csv`: prompt-specific top-`K` summaries

## Generated proposal texts and source records

The full generated Part B1-like proposal texts are not included in this public anonymous repository to avoid circulation outside the research context and possible confusion with the actual Part B1 proposals of funded projects. They will be made available upon request for research and reproducibility purposes. The source CORDIS records can be reconstructed from the identifiers and retrieval information described in the manuscript and accompanying materials.

## Notes

- `q06` denotes the overall score question.
- “Pooled” outputs aggregate across models and prompt templates.
- “Robustness” files summarize consistency across models, prompts, or proposal texts.
- “Pairwise” and “pair difference” files compare one contrast effect against another, rather than against the blind baseline alone.
- The `rank_selection` folder translates score changes into ranking and top-`K` selection consequences.

## License and Use

Code in this repository is released under the MIT License. Derived analysis outputs, including tables, figures, and other result files produced for the paper, are released under Creative Commons Attribution 4.0 International (CC BY 4.0). This means the code may be reused, modified, and redistributed under the terms of the MIT License, while the released results may be shared and adapted with appropriate attribution. Any reuse of source materials obtained from external providers remains subject to the original terms and licenses of those sources.
