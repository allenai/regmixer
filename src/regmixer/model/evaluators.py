from enum import Enum



class DownstreamEvaluatorsSmall(Enum):
    """Enum class enumerating in-loop evaluators for small models."""

    ARC_CHALLENGE="arc_challenge_test_rc_5shot"
    ARC_EASY="arc_easy_test_rc_5shot"
    HELLASWAG="hellaswag_rc_5shot" # 1K subset of HellaSwag
    WINOGRANDE="winogrande_val_rc_5shot" # Helpful after 750M-5xC scale
    CSQA="csqa_val_rc_5shot"
    PIQA="piqa_val_rc_5shot"
    SOCIALIQA="socialiqa_val_rc_5shot"

    # Too noisy to be worth tracking
    # "boolq_val_rc_5shot",
    # "openbookqa_test_rc_5shot",

    # MMLU RC
    MMLU_STEM_VAL="mmlu_stem_val_rc_5shot"
    MMLU_HUMANITIES_VAL="mmlu_humanities_val_rc_5shot"
    MMLU_SOCIAL_SCIENCES_VAL="mmlu_social_sciences_val_rc_5shot"
    MMLU_OTHER_VAL="mmlu_other_val_rc_5shot"
    MMLU_STEM_TEST="mmlu_stem_test_rc_5shot"
    MMLU_HUMANITIES_TEST="mmlu_humanities_test_rc_5shot"
    MMLU_SOCIAL_SCIENCES_TEST="mmlu_social_sciences_test_rc_5shot"
    MMLU_OTHER_TEST="mmlu_other_test_rc_5shot"

    # Gen tasks BPB
    GSM8K="gsm8k_gold_bpb_5shot"
    MINERVA_ALGEBRA="minerva_math_algebra_gold_bpb_0shot"
    MINERVA_COUNTING="minerva_math_counting_and_probability_gold_bpb_0shot"
    MINERVA_GEOMETRY="minerva_math_geometry_gold_bpb_0shot"
    MINERVA_INTERMEDIATE="minerva_math_intermediate_algebra_gold_bpb_0shot"
    MINERVA_NUMBER="minerva_math_number_theory_gold_bpb_0shot"
    MINERVA_PREALGEBRA="minerva_math_prealgebra_gold_bpb_0shot"
    MINERVA_PRECALCULUS="minerva_math_precalculus_gold_bpb_0shot"
    CODEX_HUMANEVAL="codex_humaneval_gold_bpb_0shot"
    CODEX_MBPP="codex_mbpp_gold_bpb_0shot"

    # Sanity check for MCQA ability
    COPYCOLORS="copycolors_10way"



class DownstreamEvaluators(Enum):
    """Enum class enumerating available in-loop evaluators."""

    PIQA = "piqa"
    HELLASWAG = "hellaswag"
    WINOGRANDE = "winogrande"
    OPENBOOK_QA = "openbook_qa"
    BOOLQ = "boolq"
    SCIQ = "sciq"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    COPA = "copa"
    COMMONSENSE_QA = "commonsense_qa"
    SOCIAL_IQA = "social_iqa"
    MMLU_STEM_VAR = "mmlu_stem_var"
    MMLU_HUMANITIES_VAR = "mmlu_humanities_var"
    MMLU_SOCIAL_SCIENCES_VAR = "mmlu_social_sciences_var"
    MMLU_OTHER_VAR = "mmlu_other_var"
    MMLU_STEM_MC_5SHOT = "mmlu_stem_mc_5shot"
    MMLU_HUMANITIES_MC_5SHOT = "mmlu_humanities_mc_5shot"
    MMLU_SOCIAL_SCIENCES_MC_5SHOT = "mmlu_social_sciences_mc_5shot"
    MMLU_OTHER_MC_5SHOT = "mmlu_other_mc_5shot"
    MMLU_STEM_MC_5SHOT_TEST = "mmlu_stem_mc_5shot_test"
    MMLU_HUMANITIES_MC_5SHOT_TEST = "mmlu_humanities_mc_5shot_test"
    MMLU_SOCIAL_SCIENCES_MC_5SHOT_TEST = "mmlu_social_sciences_mc_5shot_test"
    MMLU_OTHER_MC_5SHOT_TEST = "mmlu_other_mc_5shot_test"
    BASIC_ARITHMETIC = "basic_arithmetic"
    TRIVIA_QA_WIKI_PPL = "trivia_qa_wiki_ppl"
    NATURAL_QS_OPEN_PPL = "natural_qs_open_ppl"
    ARC_EASY_PPL = "arc_easy_ppl"
    # PIQA_RC_0SHOT = "piqa_rc_0shot"
    # PIQA_RC_0SHOT_BPB = "piqa_rc_0shot_bpb"
    PIQA_RC_5SHOT = "piqa_rc_5shot"
    PIQA_RC_5SHOT_BPB = "piqa_rc_5shot_bpb"
    PIQA_MC_5SHOT = "piqa_mc_5shot"
    PIQA_MC_5SHOT_BPB = "piqa_mc_5shot_bpb"
    # HELLASWAG_RC_0SHOT = "hellaswag_rc_0shot"
    # HELLASWAG_RC_0SHOT_BPB = "hellaswag_rc_0shot_bpb"
    HELLASWAG_RC_5SHOT = "hellaswag_rc_5shot"
    HELLASWAG_RC_5SHOT_BPB = "hellaswag_rc_5shot_bpb"
    HELLASWAG_MC_5SHOT = "hellaswag_mc_5shot"
    HELLASWAG_MC_5SHOT_BPB = "hellaswag_mc_5shot_bpb"
    # WINOGRANDE_RC_0SHOT = "winogrande_rc_0shot"
    # WINOGRANDE_RC_0SHOT_BPB = "winogrande_rc_0shot_bpb"
    WINOGRANDE_RC_5SHOT = "winogrande_rc_5shot"
    WINOGRANDE_RC_5SHOT_BPB = "winogrande_rc_5shot_bpb"
    WINOGRANDE_MC_5SHOT = "winogrande_mc_5shot"
    WINOGRANDE_MC_5SHOT_BPB = "winogrande_mc_5shot_bpb"
    # OPENBOOKQA_RC_0SHOT = "openbookqa_rc_0shot"
    # OPENBOOKQA_RC_0SHOT_BPB = "openbookqa_rc_0shot_bpb"
    OPENBOOKQA_RC_5SHOT = "openbookqa_rc_5shot"
    OPENBOOKQA_RC_5SHOT_BPB = "openbookqa_rc_5shot_bpb"
    OPENBOOKQA_MC_5SHOT = "openbookqa_mc_5shot"
    OPENBOOKQA_MC_5SHOT_BPB = "openbookqa_mc_5shot_bpb"
    # BOOLQ_RC_0SHOT = "boolq_rc_0shot"
    # BOOLQ_RC_0SHOT_BPB = "boolq_rc_0shot_bpb"
    BOOLQ_RC_5SHOT = "boolq_rc_5shot"
    BOOLQ_RC_5SHOT_BPB = "boolq_rc_5shot_bpb"
    BOOLQ_MC_5SHOT = "boolq_mc_5shot"
    BOOLQ_MC_5SHOT_BPB = "boolq_mc_5shot_bpb"
    # SCIQ_RC_0SHOT = "sciq_rc_0shot"
    # SCIQ_RC_0SHOT_BPB = "sciq_rc_0shot_bpb"
    # ARC_EASY_RC_0SHOT = "arc_easy_rc_0shot"
    # ARC_EASY_RC_0SHOT_BPB = "arc_easy_rc_0shot_bpb"
    ARC_EASY_RC_5SHOT = "arc_easy_rc_5shot"
    ARC_EASY_RC_5SHOT_BPB = "arc_easy_rc_5shot_bpb"
    ARC_EASY_MC_5SHOT = "arc_easy_mc_5shot"
    ARC_EASY_MC_5SHOT_BPB = "arc_easy_mc_5shot_bpb"
    # ARC_CHALLENGE_RC_0SHOT = "arc_challenge_rc_0shot"
    # ARC_CHALLENGE_RC_0SHOT_BPB = "arc_challenge_rc_0shot_bpb"
    ARC_CHALLENGE_RC_5SHOT = "arc_challenge_rc_5shot"
    ARC_CHALLENGE_RC_5SHOT_BPB = "arc_challenge_rc_5shot_bpb"
    ARC_CHALLENGE_MC_5SHOT = "arc_challenge_mc_5shot"
    ARC_CHALLENGE_MC_5SHOT_BPB = "arc_challenge_mc_5shot_bpb"
    # COPA_RC_0SHOT = "copa_rc_0shot"
    # COPA_RC_0SHOT_BPB = "copa_rc_0shot_bpb"
    # CSQA_RC_0SHOT = "csqa_rc_0shot"
    # CSQA_RC_0SHOT_BPB = "csqa_rc_0shot_bpb"
    CSQA_RC_5SHOT = "csqa_rc_5shot"
    CSQA_RC_5SHOT_BPB = "csqa_rc_5shot_bpb"
    CSQA_MC_5SHOT = "csqa_mc_5shot"
    CSQA_MC_5SHOT_BPB = "csqa_mc_5shot_bpb"
    # SOCIALIQA_RC_0SHOT = "socialiqa_rc_0shot"
    # SOCIALIQA_RC_0SHOT_BPB = "socialiqa_rc_0shot_bpb"
    SOCIALIQA_RC_5SHOT = "socialiqa_rc_5shot"
    SOCIALIQA_RC_5SHOT_BPB = "socialiqa_rc_5shot_bpb"
    SOCIALIQA_MC_5SHOT = "socialiqa_mc_5shot"
    SOCIALIQA_MC_5SHOT_BPB = "socialiqa_mc_5shot_bpb"
    MMLU_STEM_VAR_BPB = "mmlu_stem_var_bpb"
    MMLU_HUMANITIES_VAR_BPB = "mmlu_humanities_var_bpb"
    MMLU_SOCIAL_SCIENCES_VAR_BPB = "mmlu_social_sciences_var_bpb"
    MMLU_OTHER_VAR_BPB = "mmlu_other_var_bpb"
    MMLU_STEM_BPB = "mmlu_stem_bpb"
    MMLU_HUMANITIES_BPB = "mmlu_humanities_bpb"
    MMLU_SOCIAL_SCIENCES_BPB = "mmlu_social_sciences_bpb"
    MMLU_OTHER_BPB = "mmlu_other_bpb"
