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
    all_bpb_with_offline = [
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
        'basic_skills_arithmetic:rc::olmes',
        'basic_skills_coding:rc::olmes',
        'basic_skills_common_knowledge:rc::olmes',
        'basic_skills_logical_reasoning:rc::olmes',
        'basic_skills_string_operations:rc::olmes',
        'basic_skills_pattern:rc::olmes',
        'mt_mbpp:bash',
        'mt_mbpp:c',
        'mt_mbpp:cpp',
        'mt_mbpp:csharp',
        'mt_mbpp:go',
        'mt_mbpp:haskell',
        'mt_mbpp:java',
        'mt_mbpp:javascript',
        'mt_mbpp:matlab',
        'mt_mbpp:php',
        'mt_mbpp:python',
        'mt_mbpp:r',
        'mt_mbpp:ruby',
        'mt_mbpp:rust',
        'mt_mbpp:scala',
        'mt_mbpp:swift',
        'mt_mbpp:typescript',
]


# arc_easy, arc_challenge, hellaswag, MMLU macro average, 
# GSM8K is a step function, is bad 


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


class ObjectiveWeights(Enum):
    aggregated_minerva_codex = {
        WandbMetrics.minerva_math_precalculus_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_prealgebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_number_theory_gold_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_intermediate_algebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_geometry_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_counting_and_probability_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_algebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.codex_mbpp_gold_bpb_0shot.value: 0.5,
        WandbMetrics.codex_humaneval_gold_bpb_0shot.value: 0.5
    }

    aggregated_minerva_codex_mtmbpp_skills = {
        WandbMetrics.minerva_math_precalculus_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_prealgebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_number_theory_gold_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_intermediate_algebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_geometry_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_counting_and_probability_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.minerva_math_algebra_gold_bpb_0shot.value: 1.0/7,
        WandbMetrics.codex_mbpp_gold_bpb_0shot.value: 0.5,
        WandbMetrics.codex_humaneval_gold_bpb_0shot.value: 0.5,
        'basic_skills_arithmetic:rc::olmes': 1.0/6,
        'basic_skills_coding:rc::olmes': 1.0/6,
        'basic_skills_common_knowledge:rc::olmes': 1.0/6,
        'basic_skills_logical_reasoning:rc::olmes': 1.0/6,
        'basic_skills_string_operations:rc::olmes': 1.0/6,
        'basic_skills_pattern:rc::olmes': 1.0/6,
        'mt_mbpp:bash': 1.0/17,
        'mt_mbpp:c': 1.0/17,
        'mt_mbpp:cpp': 1.0/17,
        'mt_mbpp:csharp': 1.0/17,
        'mt_mbpp:go': 1.0/17,
        'mt_mbpp:haskell': 1.0/17,
        'mt_mbpp:java': 1.0/17,
        'mt_mbpp:javascript': 1.0/17,
        'mt_mbpp:matlab': 1.0/17,
        'mt_mbpp:php': 1.0/17,
        'mt_mbpp:python': 1.0/17,
        'mt_mbpp:r': 1.0/17,
        'mt_mbpp:ruby': 1.0/17,
        'mt_mbpp:rust': 1.0/17,
        'mt_mbpp:scala': 1.0/17,
        'mt_mbpp:swift': 1.0/17,
        'mt_mbpp:typescript': 1.0/17,
    }

    correlation_weighting = {
        'eval/downstream/codex_mbpp_gold_bpb_0shot (BPB)': 0.007478495932052726,
        'mt_mbpp:r': 0.007581472807703164,
        'mt_mbpp:cpp': 0.007596164255702139,
        'mt_mbpp:swift': 0.007646357091696129,
        'mt_mbpp:javascript': 0.007728457152801651,
        'mt_mbpp:python': 0.00775014680758216,
        'mt_mbpp:c': 0.007827819716306252,
        'mt_mbpp:csharp': 0.007848598954150713,
        'mt_mbpp:go': 0.007853969825648928,
        'mt_mbpp:php': 0.007981238305130157,
        'mt_mbpp:matlab': 0.008012781032538609,
        'mt_mbpp:java': 0.008019525474139278,
        'mt_mbpp:typescript': 0.008060229157312172,
        'mt_mbpp:scala': 0.008107169877202051,
        'mt_mbpp:ruby': 0.008115599601805749,
        'eval/downstream/codex_humaneval_gold_bpb_0shot (BPB)': 0.008132727142094193,
        'mt_mbpp:rust': 0.008322200939272236,
        'mt_mbpp:bash': 0.008494776036513139,
        'mt_mbpp:haskell': 0.00913328301431387,
        'basic_skills_coding:rc::olmes': 0.00960003772776684,
        'eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB)': 0.013086105002452258,
        'eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB)': 0.014189878412247324,
        'eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB)': 0.017135627292711984,
        'eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB)': 0.01771918909431664,
        'eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB)': 0.018390167179091803,
        'eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB)': 0.019011313843287178,
        'eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB)': 0.019110868314201397,
        'eval/downstream/gsm8k_gold_bpb_5shot (BPB)': 0.02461692821657061,
        'basic_skills_string_operations:rc::olmes': 0.025344216960002403,
        'basic_skills_arithmetic:rc::olmes': 0.02681478223856301,
        'basic_skills_logical_reasoning:rc::olmes': 0.030315063906300472,
        'eval/downstream/mmlu_humanities_test_rc_5shot (BPB)': 0.03172484348282645,
        'eval/downstream/csqa_val_rc_5shot (BPB)': 0.03308222817301801,
        'eval/downstream/mmlu_stem_test_rc_5shot (BPB)': 0.03566947155637326,
        'eval/downstream/piqa_val_rc_5shot (BPB)': 0.03758628295512897,
        'eval/downstream/socialiqa_val_rc_5shot (BPB)': 0.0388614043357008,
        'eval/downstream/winogrande_val_rc_5shot (BPB)': 0.044233455753328585,
        'eval/downstream/hellaswag_rc_5shot (BPB)': 0.04618827047588011,
        'eval/downstream/arc_easy_test_rc_5shot (BPB)': 0.049269604993780164,
        'basic_skills_pattern:rc::olmes': 0.04993743233941554,
        'eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB)': 0.05389662712046272,
        'eval/downstream/arc_challenge_test_rc_5shot (BPB)': 0.054803671370572585,
        'basic_skills_common_knowledge:rc::olmes': 0.05919216023553791,
        'eval/downstream/mmlu_other_test_rc_5shot (BPB)': 0.07852935589649764
    }

    correlation_weighting_dense = {
        'mt_mbpp:python': 0.005492809755171467,
        'eval/downstream/codex_mbpp_gold_bpb_0shot (BPB)': 0.005498468787973884,
        'eval/downstream/codex_humaneval_gold_bpb_0shot (BPB)': 0.005605804252926776,
        'mt_mbpp:matlab': 0.005901591779451514,
        'mt_mbpp:r': 0.005930836562291129,
        'mt_mbpp:swift': 0.0060535096926560405,
        'basic_skills_coding:rc::olmes': 0.006053807791741199,
        'mt_mbpp:javascript': 0.006115871536504707,
        'mt_mbpp:c': 0.006136563136743174,
        'mt_mbpp:typescript': 0.006153839852673626,
        'mt_mbpp:go': 0.006244647757799878,
        'mt_mbpp:ruby': 0.006279992105782575,
        'mt_mbpp:cpp': 0.006370922125769096,
        'mt_mbpp:php': 0.00637767504193833,
        'mt_mbpp:scala': 0.006382800234035382,
        'mt_mbpp:bash': 0.00641233474591976,
        'mt_mbpp:java': 0.006440366585933535,
        'mt_mbpp:csharp': 0.006444772015427053,
        'mt_mbpp:rust': 0.0066388504766329,
        'mt_mbpp:haskell': 0.0067945609547511564,
        'eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB)': 0.008089007265185737,
        'eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB)': 0.010498243109255855,
        'eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB)': 0.012503482306143818,
        'eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB)': 0.012553939364243502,
        'eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB)': 0.012958134754943186,
        'eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB)': 0.013843798408037388,
        'eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB)': 0.014100621410648306,
        'basic_skills_string_operations:rc::olmes': 0.014586231437501618,
        'basic_skills_arithmetic:rc::olmes': 0.02113909751069871,
        'eval/downstream/gsm8k_gold_bpb_5shot (BPB)': 0.021433016358519588,
        'eval/downstream/mmlu_stem_test_rc_5shot (BPB)': 0.022445223073103017,
        'eval/downstream/socialiqa_val_rc_5shot (BPB)': 0.02461608613346242,
        'eval/downstream/csqa_val_rc_5shot (BPB)': 0.024932881655192588,
        'eval/downstream/mmlu_humanities_test_rc_5shot (BPB)': 0.03238049584760073,
        'eval/downstream/arc_challenge_test_rc_5shot (BPB)': 0.033324964796330646,
        'basic_skills_logical_reasoning:rc::olmes': 0.03514484757997812,
        'basic_skills_pattern:rc::olmes': 0.0407818296224541,
        'eval/downstream/arc_easy_test_rc_5shot (BPB)': 0.04361179321474073,
        'eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB)': 0.04691008975961886,
        'eval/downstream/mmlu_other_test_rc_5shot (BPB)': 0.054092532341680176,
        'basic_skills_common_knowledge:rc::olmes': 0.0722147872861702,
        'eval/downstream/piqa_val_rc_5shot (BPB)': 0.09061994925864167,
        'eval/downstream/winogrande_val_rc_5shot (BPB)': 0.0914827214381891,
        'eval/downstream/hellaswag_rc_5shot (BPB)': 0.1224062008755367
    }


    mbpp_dummy = {
        #'eval/downstream/codex_mbpp_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/codex_humaneval_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/gsm8k_gold_bpb_5shot (BPB)': 0,
        'eval/downstream/mmlu_humanities_test_rc_5shot (BPB)': 0,
        'eval/downstream/csqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_stem_test_rc_5shot (BPB)': 0,
        'eval/downstream/piqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/socialiqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/winogrande_val_rc_5shot (BPB)': 0,
        'eval/downstream/hellaswag_rc_5shot (BPB)': 0,
        'eval/downstream/arc_easy_test_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB)': 0,
        'eval/downstream/arc_challenge_test_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_other_test_rc_5shot (BPB)': 0,
    }

    minerva_dummy = {
        'eval/downstream/codex_mbpp_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/codex_humaneval_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB)': 0,
        #'eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB)': 0,
        'eval/downstream/gsm8k_gold_bpb_5shot (BPB)': 0,
        'eval/downstream/mmlu_humanities_test_rc_5shot (BPB)': 0,
        'eval/downstream/csqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_stem_test_rc_5shot (BPB)': 0,
        'eval/downstream/piqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/socialiqa_val_rc_5shot (BPB)': 0,
        'eval/downstream/winogrande_val_rc_5shot (BPB)': 0,
        'eval/downstream/hellaswag_rc_5shot (BPB)': 0,
        'eval/downstream/arc_easy_test_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB)': 0,
        'eval/downstream/arc_challenge_test_rc_5shot (BPB)': 0,
        'eval/downstream/mmlu_other_test_rc_5shot (BPB)': 0,

    }