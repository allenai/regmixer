from enum import Enum

BASE_METRICS_PATH = "ai2-llm/regmixer"


class WandbMetrics(Enum):
    arc_challenge_len_norm = "eval/downstream/arc_challenge (length-normalized accuracy)"
    arc_challenge_mc_5shot_accuracy = "eval/downstream/arc_challenge_mc_5shot (accuracy)"
    arc_challenge_mc_5shot_bpb = "eval/downstream/arc_challenge_mc_5shot_bpb (BPB)"
    arc_challenge_rc_5shot_bpb = "eval/downstream/arc_challenge_rc_5shot_bpb (BPB)"
    arc_challenge_rc_5shot_len_norm = (
        "eval/downstream/arc_challenge_rc_5shot (length-normalized accuracy)"
    )
    arc_challenge_test_rc_5shot_bpb = "eval/downstream/arc_challenge_test_rc_5shot (BPB)"
    arc_easy_test_rc_5shot_bpb = "eval/downstream/arc_easy_test_rc_5shot (BPB)"
    arc_easy_accuracy = "eval/downstream/arc_easy (accuracy)"
    arc_easy_mc_5shot_accuracy = "eval/downstream/arc_easy_mc_5shot (accuracy)"
    arc_easy_mc_5shot_bpb = "eval/downstream/arc_easy_mc_5shot_bpb (BPB)"
    arc_easy_ppl_ce_loss = "eval/downstream/arc_easy_ppl (CE loss)"
    arc_easy_rc_5shot_accuracy = "eval/downstream/arc_easy_rc_5shot (accuracy)"
    arc_easy_rc_5shot_bpb = "eval/downstream/arc_easy_rc_5shot_bpb (BPB)"
    basic_arithmetic_accuracy = "eval/downstream/basic_arithmetic (accuracy)"
    boolq_accuracy = "eval/downstream/boolq (accuracy)"
    boolq_mc_5shot_accuracy = "eval/downstream/boolq_mc_5shot (accuracy)"
    boolq_mc_5shot_bpb = "eval/downstream/boolq_mc_5shot_bpb (BPB)"
    boolq_rc_5shot_accuracy = "eval/downstream/boolq_rc_5shot (accuracy)"
    boolq_rc_5shot_bpb = "eval/downstream/boolq_rc_5shot_bpb (BPB)"
    c4_en_validation_ce_loss = "eval/lm/c4_en-validation/CE loss"
    c4_en_validation_ppl = "eval/lm/c4_en-validation/PPL"
    codex_humaneval_gold_bpb_0shot = "eval/downstream/codex_humaneval_gold_bpb_0shot (BPB)"
    codex_mbpp_gold_bpb_0shot = "eval/downstream/codex_mbpp_gold_bpb_0shot (BPB)"
    commonsense_qa_len_norm = "eval/downstream/commonsense_qa (length-normalized accuracy)"
    copa_accuracy = "eval/downstream/copa (accuracy)"
    csqa_mc_5shot_accuracy = "eval/downstream/csqa_mc_5shot (accuracy)"
    csqa_mc_5shot_bpb = "eval/downstream/csqa_mc_5shot_bpb (BPB)"
    csqa_rc_5shot_bpb = "eval/downstream/csqa_rc_5shot_bpb (BPB)"
    csqa_rc_5shot_len_norm = "eval/downstream/csqa_rc_5shot (length-normalized accuracy)"
    csqa_val_rc_5shot_bpb = "eval/downstream/csqa_val_rc_5shot (BPB)"
    dolma_books_validation_ce_loss = "eval/lm/dolma_books-validation/CE loss"
    dolma_books_validation_ppl = "eval/lm/dolma_books-validation/PPL"
    dolma_common_crawl_validation_ce_loss = "eval/lm/dolma_common-crawl-validation/CE loss"
    dolma_common_crawl_validation_ppl = "eval/lm/dolma_common-crawl-validation/PPL"
    dolma_pes2o_validation_ce_loss = "eval/lm/dolma_pes2o-validation/CE loss"
    dolma_pes2o_validation_ppl = "eval/lm/dolma_pes2o-validation/PPL"
    dolma_reddit_validation_ce_loss = "eval/lm/dolma_reddit-validation/CE loss"
    dolma_reddit_validation_ppl = "eval/lm/dolma_reddit-validation/PPL"
    dolma_stack_validation_ce_loss = "eval/lm/dolma_stack-validation/CE loss"
    dolma_stack_validation_ppl = "eval/lm/dolma_stack-validation/PPL"
    dolma_wiki_validation_ce_loss = "eval/lm/dolma_wiki-validation/CE loss"
    dolma_wiki_validation_ppl = "eval/lm/dolma_wiki-validation/PPL"
    gsm8k_gold_bpb_5shot = "eval/downstream/gsm8k_gold_bpb_5shot (BPB)"
    hellaswag_len_norm = "eval/downstream/hellaswag (length-normalized accuracy)"
    hellaswag_len_norm_accuracy = "eval/downstream/hellaswag (length-normalized accuracy)"
    hellaswag_mc_5shot_accuracy = "eval/downstream/hellaswag_mc_5shot (accuracy)"
    hellaswag_mc_5shot_bpb = "eval/downstream/hellaswag_mc_5shot_bpb (BPB)"
    hellaswag_rc_5shot_bpb = "eval/downstream/hellaswag_rc_5shot (BPB)"
    hellaswag_rc_5shot_bpb_v2 = "eval/downstream/hellaswag_rc_5shot_bpb (BPB v2)"
    hellaswag_rc_5shot_len_norm = "eval/downstream/hellaswag_rc_5shot (length-normalized accuracy)"
    ice_validation_ce_loss = "eval/lm/ice-validation/CE loss"
    ice_validation_ppl = "eval/lm/ice-validation/PPL"
    m2d2_s2orc_validation_ce_loss = "eval/lm/m2d2_s2orc-validation/CE loss"
    m2d2_s2orc_validation_ppl = "eval/lm/m2d2_s2orc-validation/PPL"
    minerva_math_precalculus_gold_bpb_0shot = "eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB)"
    minerva_math_prealgebra_gold_bpb_0shot = "eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB)"
    minerva_math_number_theory_gold_0shot = "eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB)"
    minerva_math_intermediate_algebra_gold_bpb_0shot = "eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB)"
    minerva_math_geometry_gold_bpb_0shot = "eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB)"
    minerva_math_counting_and_probability_gold_bpb_0shot = "eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB)"
    minerva_math_algebra_gold_bpb_0shot = "eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB)"
    mmlu_humanities_bpb = "eval/downstream/mmlu_humanities_bpb (BPB)"
    mmlu_humanities_test_rc_5shot_bpb = "eval/downstream/mmlu_humanities_test_rc_5shot (BPB)"
    mmlu_humanities_mc_5shot_len_norm = (
        "eval/downstream/mmlu_humanities_mc_5shot (length-normalized accuracy)"
    )
    mmlu_humanities_mc_5shot_test_len_norm = (
        "eval/downstream/mmlu_humanities_mc_5shot_test (length-normalized accuracy)"
    )
    mmlu_humanities_var_bpb = "eval/downstream/mmlu_humanities_var_bpb (BPB)"
    mmlu_humanities_var_len_norm = (
        "eval/downstream/mmlu_humanities_var (length-normalized accuracy)"
    )
    mmlu_other_bpb = "eval/downstream/mmlu_other_bpb (BPB)"
    mmlu_other_test_rc_5shot_bpb = "eval/downstream/mmlu_other_test_rc_5shot (BPB)"
    mmlu_other_mc_5shot_len_norm = (
        "eval/downstream/mmlu_other_mc_5shot (length-normalized accuracy)"
    )
    mmlu_other_mc_5shot_test_len_norm = (
        "eval/downstream/mmlu_other_mc_5shot_test (length-normalized accuracy)"
    )
    mmlu_other_var_bpb = "eval/downstream/mmlu_other_var_bpb (BPB)"
    mmlu_other_var_len_norm = "eval/downstream/mmlu_other_var (length-normalized accuracy)"
    mmlu_social_sciences_bpb = "eval/downstream/mmlu_social_sciences_bpb (BPB)"
    mmlu_social_sciences_test_rc_5shot_bpb = "eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB)"
    mmlu_social_sciences_mc_5shot_len_norm = (
        "eval/downstream/mmlu_social_sciences_mc_5shot (length-normalized accuracy)"
    )
    mmlu_social_sciences_mc_5shot_test_len_norm = (
        "eval/downstream/mmlu_social_sciences_mc_5shot_test (length-normalized accuracy)"
    )
    mmlu_social_sciences_var_bpb = "eval/downstream/mmlu_social_sciences_var_bpb (BPB)"
    mmlu_social_sciences_var_len_norm = (
        "eval/downstream/mmlu_social_sciences_var (length-normalized accuracy)"
    )
    mmlu_stem_bpb = "eval/downstream/mmlu_stem_bpb (BPB)"
    mmlu_stem_test_rc_5shot_bpb = "eval/downstream/mmlu_stem_test_rc_5shot (BPB)"
    mmlu_stem_mc_5shot_len_norm = "eval/downstream/mmlu_stem_mc_5shot (length-normalized accuracy)"
    mmlu_stem_mc_5shot_test_len_norm = (
        "eval/downstream/mmlu_stem_mc_5shot_test (length-normalized accuracy)"
    )
    mmlu_stem_var_bpb = "eval/downstream/mmlu_stem_var_bpb (BPB)"
    mmlu_stem_var_len_norm = "eval/downstream/mmlu_stem_var (length-normalized accuracy)"
    natural_qs_open_ppl_ce_loss = "eval/downstream/natural_qs_open_ppl (CE loss)"
    openbook_qa_len_norm = "eval/downstream/openbook_qa (length-normalized accuracy)"
    openbookqa_mc_5shot_accuracy = "eval/downstream/openbookqa_mc_5shot (accuracy)"
    openbookqa_mc_5shot_bpb = "eval/downstream/openbookqa_mc_5shot_bpb (BPB)"
    openbookqa_rc_5shot_bpb = "eval/downstream/openbookqa_rc_5shot_bpb (BPB)"
    openbookqa_rc_5shot_len_norm = (
        "eval/downstream/openbookqa_rc_5shot (length-normalized accuracy)"
    )
    pile_validation_ce_loss = "eval/lm/pile-validation/CE loss"
    pile_validation_ppl = "eval/lm/pile-validation/PPL"
    piqa_len_norm = "eval/downstream/piqa (length-normalized accuracy)"
    piqa_mc_5shot_accuracy = "eval/downstream/piqa_mc_5shot (accuracy)"
    piqa_mc_5shot_bpb = "eval/downstream/piqa_mc_5shot_bpb (BPB)"
    piqa_rc_5shot_bpb = "eval/downstream/piqa_rc_5shot_bpb (BPB)"
    piqa_rc_5shot_len_norm = "eval/downstream/piqa_rc_5shot (length-normalized accuracy)"
    piqa_val_rc_5shot_bpb = "eval/downstream/piqa_val_rc_5shot (BPB)"
    sciq_accuracy = "eval/downstream/sciq (accuracy)"
    social_iqa_len_norm = "eval/downstream/social_iqa (length-normalized accuracy)"
    socialiqa_mc_5shot_accuracy = "eval/downstream/socialiqa_mc_5shot (accuracy)"
    socialiqa_mc_5shot_bpb = "eval/downstream/socialiqa_mc_5shot_bpb (BPB)"
    socialiqa_rc_5shot_bpb = "eval/downstream/socialiqa_rc_5shot_bpb (BPB)"
    socialiqa_val_rc_5shot_bpb = "eval/downstream/socialiqa_val_rc_5shot (BPB)"
    socialiqa_rc_5shot_len_norm = "eval/downstream/socialiqa_rc_5shot (length-normalized accuracy)"
    throughput_device_bps = "throughput/device/BPS"
    train_ppl = "train/PPL"
    train_loss = "train/CE loss"
    trivia_qa_wiki_ppl_ce_loss = "eval/downstream/trivia_qa_wiki_ppl (CE loss)"
    wikitext_103_validation_ce_loss = "eval/lm/wikitext_103-validation/CE loss"
    wikitext_103_validation_ppl = "eval/lm/wikitext_103-validation/PPL"
    winogrande_accuracy = "eval/downstream/winogrande (accuracy)"
    winogrande_mc_5shot_accuracy = "eval/downstream/winogrande_mc_5shot (accuracy)"
    winogrande_mc_5shot_bpb = "eval/downstream/winogrande_mc_5shot_bpb (BPB)"
    winogrande_rc_5shot_accuracy = "eval/downstream/winogrande_rc_5shot (accuracy)"
    winogrande_val_rc_5shot_bpb = "eval/downstream/winogrande_val_rc_5shot (BPB)"


class GroupedWandbMetrics(Enum):
    val_loss = [
        WandbMetrics.wikitext_103_validation_ce_loss.value,
        WandbMetrics.pile_validation_ce_loss.value,
        WandbMetrics.m2d2_s2orc_validation_ce_loss.value,
        WandbMetrics.ice_validation_ce_loss.value,
        WandbMetrics.dolma_wiki_validation_ce_loss.value,
        WandbMetrics.dolma_stack_validation_ce_loss.value,
        WandbMetrics.dolma_reddit_validation_ce_loss.value,
        WandbMetrics.dolma_pes2o_validation_ce_loss.value,
        WandbMetrics.dolma_common_crawl_validation_ce_loss.value,
        WandbMetrics.dolma_books_validation_ce_loss.value,
        WandbMetrics.c4_en_validation_ce_loss.value,
    ]
    books = [
        WandbMetrics.dolma_books_validation_ce_loss.value
    ]
    c4 = [
        WandbMetrics.c4_en_validation_ce_loss.value
    ]
    hellaswag = [
        WandbMetrics.hellaswag_rc_5shot_bpb.value,
    ]
    hellaswag_v2 = [
        WandbMetrics.hellaswag_rc_5shot_bpb_v2.value,
    ]
    winogrande = [
        WandbMetrics.winogrande_val_rc_5shot_bpb.value,
    ]
    arc_challenge = [
        WandbMetrics.arc_challenge_len_norm.value,
        WandbMetrics.arc_challenge_mc_5shot_accuracy.value,
        WandbMetrics.arc_challenge_rc_5shot_len_norm.value,
    ]
    arc_challenge_bpb = [
        WandbMetrics.arc_challenge_mc_5shot_bpb.value,
        WandbMetrics.arc_challenge_rc_5shot_bpb.value,
    ]
    arc_easy = [
        WandbMetrics.arc_easy_accuracy.value,
        WandbMetrics.arc_easy_mc_5shot_accuracy.value,
        WandbMetrics.arc_easy_ppl_ce_loss.value,
        WandbMetrics.arc_easy_rc_5shot_accuracy.value,
    ]
    arc_easy_new = [
        WandbMetrics.arc_easy_test_rc_5shot_bpb.value
    ]
    arc_easy_bpb = [
        WandbMetrics.arc_easy_mc_5shot_bpb.value,
        WandbMetrics.arc_easy_rc_5shot_bpb.value,
    ]
    mmlu_bpb = [
        WandbMetrics.mmlu_humanities_bpb.value,
        WandbMetrics.mmlu_humanities_var_bpb.value,
        WandbMetrics.mmlu_other_bpb.value,
        WandbMetrics.mmlu_other_var_bpb.value,
        WandbMetrics.mmlu_social_sciences_bpb.value,
        WandbMetrics.mmlu_social_sciences_var_bpb.value,
        WandbMetrics.mmlu_stem_bpb.value,
        WandbMetrics.mmlu_stem_var_bpb.value,
    ]
    mmlu_bpb_new = [
        WandbMetrics.mmlu_humanities_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_other_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_social_sciences_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_stem_test_rc_5shot_bpb.value,
    ]
    mmlu_len_norm = [
        WandbMetrics.mmlu_humanities_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_humanities_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_humanities_var_len_norm.value,
        WandbMetrics.mmlu_other_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_other_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_other_var_len_norm.value,
        WandbMetrics.mmlu_social_sciences_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_social_sciences_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_social_sciences_var_len_norm.value,
        WandbMetrics.mmlu_stem_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_stem_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_stem_var_len_norm.value,
    ]
    few_shot = [
        WandbMetrics.arc_challenge_mc_5shot_accuracy.value,
        WandbMetrics.arc_challenge_rc_5shot_len_norm.value,
        WandbMetrics.arc_easy_mc_5shot_accuracy.value,
        WandbMetrics.arc_easy_rc_5shot_accuracy.value,
        WandbMetrics.boolq_mc_5shot_accuracy.value,
        WandbMetrics.boolq_rc_5shot_accuracy.value,
        WandbMetrics.csqa_mc_5shot_accuracy.value,
        WandbMetrics.csqa_rc_5shot_len_norm.value,
        WandbMetrics.hellaswag_mc_5shot_accuracy.value,
        WandbMetrics.hellaswag_rc_5shot_len_norm.value,
        WandbMetrics.mmlu_humanities_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_humanities_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_other_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_other_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_social_sciences_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_social_sciences_mc_5shot_test_len_norm.value,
        WandbMetrics.mmlu_stem_mc_5shot_len_norm.value,
        WandbMetrics.mmlu_stem_mc_5shot_test_len_norm.value,
        WandbMetrics.openbookqa_mc_5shot_accuracy.value,
        WandbMetrics.openbookqa_rc_5shot_len_norm.value,
        WandbMetrics.piqa_mc_5shot_accuracy.value,
        WandbMetrics.piqa_rc_5shot_len_norm.value,
        WandbMetrics.socialiqa_mc_5shot_accuracy.value,
        WandbMetrics.socialiqa_rc_5shot_len_norm.value,
        WandbMetrics.winogrande_mc_5shot_accuracy.value,
        WandbMetrics.winogrande_rc_5shot_accuracy.value,
    ]
    few_shot_bpb_mc = [
        WandbMetrics.arc_challenge_mc_5shot_bpb.value,
        WandbMetrics.arc_easy_mc_5shot_bpb.value,
        WandbMetrics.boolq_mc_5shot_bpb.value,
        WandbMetrics.csqa_mc_5shot_bpb.value,
        WandbMetrics.hellaswag_mc_5shot_bpb.value,
        WandbMetrics.openbookqa_mc_5shot_bpb.value,
        WandbMetrics.piqa_mc_5shot_bpb.value,
        WandbMetrics.socialiqa_mc_5shot_bpb.value,
        WandbMetrics.winogrande_mc_5shot_bpb.value,
    ]
    few_shot_bpb_rc = [
        WandbMetrics.arc_challenge_rc_5shot_bpb.value,
        WandbMetrics.arc_easy_rc_5shot_bpb.value,
        WandbMetrics.boolq_rc_5shot_bpb.value,
        WandbMetrics.csqa_rc_5shot_bpb.value,
        WandbMetrics.hellaswag_rc_5shot_bpb.value,
        WandbMetrics.openbookqa_rc_5shot_bpb.value,
        WandbMetrics.piqa_rc_5shot_bpb.value,
        WandbMetrics.socialiqa_rc_5shot_bpb.value,
        WandbMetrics.winogrande_val_rc_5shot_bpb.value,
    ]
    validation_loss = [
        WandbMetrics.c4_en_validation_ce_loss.value,
        WandbMetrics.dolma_books_validation_ce_loss.value,
        WandbMetrics.dolma_common_crawl_validation_ce_loss.value,
        WandbMetrics.dolma_pes2o_validation_ce_loss.value,
        WandbMetrics.dolma_reddit_validation_ce_loss.value,
        WandbMetrics.dolma_stack_validation_ce_loss.value,
        WandbMetrics.dolma_wiki_validation_ce_loss.value,
        WandbMetrics.ice_validation_ce_loss.value,
        WandbMetrics.m2d2_s2orc_validation_ce_loss.value,
        WandbMetrics.pile_validation_ce_loss.value,
    ]
    train_loss = [WandbMetrics.train_loss.value]
    all_metrics = [metric.value for metric in WandbMetrics]
    all_bpb = [
        WandbMetrics.mmlu_social_sciences_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_humanities_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_other_test_rc_5shot_bpb.value,
        WandbMetrics.mmlu_stem_test_rc_5shot_bpb.value,
        WandbMetrics.winogrande_val_rc_5shot_bpb.value,
        WandbMetrics.socialiqa_val_rc_5shot_bpb.value,
        WandbMetrics.piqa_val_rc_5shot_bpb.value,
        WandbMetrics.minerva_math_algebra_gold_bpb_0shot.value,
        WandbMetrics.minerva_math_counting_and_probability_gold_bpb_0shot.value,
        WandbMetrics.minerva_math_geometry_gold_bpb_0shot.value,
        WandbMetrics.minerva_math_intermediate_algebra_gold_bpb_0shot.value,
        WandbMetrics.minerva_math_number_theory_gold_0shot.value,
        WandbMetrics.minerva_math_prealgebra_gold_bpb_0shot.value,
        WandbMetrics.minerva_math_precalculus_gold_bpb_0shot.value,
        WandbMetrics.gsm8k_gold_bpb_5shot.value,
        WandbMetrics.hellaswag_rc_5shot_bpb.value, 
        WandbMetrics.csqa_val_rc_5shot_bpb.value,
        WandbMetrics.codex_mbpp_gold_bpb_0shot.value,
        WandbMetrics.codex_humaneval_gold_bpb_0shot.value,
        WandbMetrics.arc_easy_test_rc_5shot_bpb.value,
        WandbMetrics.arc_challenge_test_rc_5shot_bpb.value,
    ]


AUS_CLUSTERS = "ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale,ai2/neptune-cirrascale"
GOOG_CLUSTERS = "ai2/augusta-google-1"
CLUSTERS = f"{AUS_CLUSTERS},{GOOG_CLUSTERS}"
ALL_TASK_GROUPS_OPTIONS = ["mmlu", "core", "tuluish", "gen"]
ALL_FORMAT_OPTIONS = ["mc", "rc"]
ALL_MMLU_TASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]
ALL_TULUISH_TASKS = [
    "gsm8k::olmo1",
    "drop::olmes",
    "minerva_math_algebra::llama3",
    "gpqa::llama3",
    "squad2::llama3",
    "squad::olmes",
    "drop::llama3",
    "naturalqs::olmes",
    "minerva_math_counting_and_probability::llama3",
    "minerva_math_geometry::llama3",
    "minerva_math_intermediate_algebra::llama3",
    "minerva_math_number_theory::llama3",
    "minerva_math_prealgebra::llama3",
    "minerva_math_precalculus::llama3",
    "bbh_boolean_expressions:cot::none",
    "bbh_causal_judgement:cot::none",
    "bbh_date_understanding:cot::none",
    "bbh_disambiguation_qa:cot::none",
    "bbh_dyck_languages:cot::none",
    "bbh_formal_fallacies:cot::none",
    "bbh_geometric_shapes:cot::none",
    "bbh_hyperbaton:cot::none",
    "bbh_logical_deduction_five_objects:cot::none",
    "bbh_logical_deduction_seven_objects:cot::none",
    "bbh_logical_deduction_three_objects:cot::none",
    "bbh_movie_recommendation:cot::none",
    "bbh_multistep_arithmetic_two:cot::none",
    "bbh_navigate:cot::none",
    "bbh_object_counting:cot::none",
    "bbh_penguins_in_a_table:cot::none",
    "bbh_reasoning_about_colored_objects:cot::none",
    "bbh_ruin_names:cot::none",
    "bbh_salient_translation_error_detection:cot::none",
    "bbh_snarks:cot::none",
    "bbh_sports_understanding:cot::none",
    "bbh_temporal_sequences:cot::none",
    "bbh_tracking_shuffled_objects_five_objects:cot::none",
    "bbh_tracking_shuffled_objects_seven_objects:cot::none",
    "bbh_tracking_shuffled_objects_three_objects:cot::none",
    "bbh_web_of_lies:cot::none",
    "bbh_word_sorting:cot::non",
    "truthfulqa::olmo1",
]
ALL_CORE_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
]
ALL_GEN_TASKS = [
    "coqa::olmes",
    "squad::olmes",
    "jeopardy::olmes",
    "naturalqs::olmes",
    "drop::olmes",
    "gsm8k::olmo1",
]
