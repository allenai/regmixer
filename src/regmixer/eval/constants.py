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
    commonsense_qa_len_norm = "eval/downstream/commonsense_qa (length-normalized accuracy)"
    copa_accuracy = "eval/downstream/copa (accuracy)"
    csqa_mc_5shot_accuracy = "eval/downstream/csqa_mc_5shot (accuracy)"
    csqa_mc_5shot_bpb = "eval/downstream/csqa_mc_5shot_bpb (BPB)"
    csqa_rc_5shot_bpb = "eval/downstream/csqa_rc_5shot_bpb (BPB)"
    csqa_rc_5shot_len_norm = "eval/downstream/csqa_rc_5shot (length-normalized accuracy)"
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
    hellaswag_len_norm = "eval/downstream/hellaswag (length-normalized accuracy)"
    hellaswag_len_norm_accuracy = "eval/downstream/hellaswag (length-normalized accuracy)"
    hellaswag_mc_5shot_accuracy = "eval/downstream/hellaswag_mc_5shot (accuracy)"
    hellaswag_mc_5shot_bpb = "eval/downstream/hellaswag_mc_5shot_bpb (BPB)"
    hellaswag_rc_5shot_bpb = "eval/downstream/hellaswag_rc_5shot_bpb (BPB)"
    hellaswag_rc_5shot_len_norm = "eval/downstream/hellaswag_rc_5shot (length-normalized accuracy)"
    ice_validation_ce_loss = "eval/lm/ice-validation/CE loss"
    ice_validation_ppl = "eval/lm/ice-validation/PPL"
    m2d2_s2orc_validation_ce_loss = "eval/lm/m2d2_s2orc-validation/CE loss"
    m2d2_s2orc_validation_ppl = "eval/lm/m2d2_s2orc-validation/PPL"
    mmlu_humanities_bpb = "eval/downstream/mmlu_humanities_bpb (BPB)"
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
    mmlu_other_mc_5shot_len_norm = (
        "eval/downstream/mmlu_other_mc_5shot (length-normalized accuracy)"
    )
    mmlu_other_mc_5shot_test_len_norm = (
        "eval/downstream/mmlu_other_mc_5shot_test (length-normalized accuracy)"
    )
    mmlu_other_var_bpb = "eval/downstream/mmlu_other_var_bpb (BPB)"
    mmlu_other_var_len_norm = "eval/downstream/mmlu_other_var (length-normalized accuracy)"
    mmlu_social_sciences_bpb = "eval/downstream/mmlu_social_sciences_bpb (BPB)"
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
    sciq_accuracy = "eval/downstream/sciq (accuracy)"
    social_iqa_len_norm = "eval/downstream/social_iqa (length-normalized accuracy)"
    socialiqa_mc_5shot_accuracy = "eval/downstream/socialiqa_mc_5shot (accuracy)"
    socialiqa_mc_5shot_bpb = "eval/downstream/socialiqa_mc_5shot_bpb (BPB)"
    socialiqa_rc_5shot_bpb = "eval/downstream/socialiqa_rc_5shot_bpb (BPB)"
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
    winogrande_rc_5shot_bpb = "eval/downstream/winogrande_rc_5shot_bpb (BPB)"


class GroupedWandbMetrics(Enum):
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
        WandbMetrics.winogrande_rc_5shot_bpb.value,
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
