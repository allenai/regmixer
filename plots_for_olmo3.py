# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "seaborn",
# ]
# ///


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_weights(
    prior: dict[str, float],
    prediction: dict[str, float],
    save_name: str
):

    columns = list(prediction.keys())

    preds = list(prediction.values())

    df = pd.DataFrame(
        data=np.concatenate(
            [
                np.array([list(prior.values())]),
                np.array([preds]),
            ],
            axis=0,
        ),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Natural"] + ["Ours"]) * len(columns)

    plt.rc("axes", unicode_minus=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
        }
    )

    _, ax = plt.subplots(figsize=(12, 10), layout="compressed")
    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(axis="x", labelrotation=90)

    pallette = {
        "Natural": "#105257",
        "Ours": "#F0529C",
    }

    df_sorted = df[df["type"] == "Natural"].sort_values(by="value", ascending=False)
    df["variable"] = pd.Categorical(df["variable"], categories=df_sorted["variable"], ordered=True)
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pallette, ax=ax)

    ax.legend(
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 18,
        },
        handlelength=0.4,
        ncol=3,
    )

    ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)
    ax.set_ylim(0, 0.4)

    ax.set_xlabel(
        "Domain",
        fontdict={
            "size": 26,
        },
    )
    ax.set_ylabel(
        "Weight",
        fontdict={
            "size": 26,
        },
    )

    plt.savefig(
        f"{save_name}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )



def plot_pareto(diff: dict[str, float], save_name: str):
    diff = {k.split("/")[-1].split(" ")[0].split("::")[0] : v for k, v in diff.items()}
    diff = pd.Series(diff)
    colors = ['green' if val > 0 else 'red' for val in diff]
    plt.figure(figsize=(10, 8))
    diff.plot(kind='bar', color=colors)
    plt.title(f'Per-task improvement over natural distribution')
    plt.ylabel('BPB Difference (higher is better)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        f"{save_name}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )




# note to self: used "all_bpb_with_offline_extended_v2" metrics for optimization
dclm_natural = {'adult_content': 0.013552015640753739,
 'art_and_design': 0.014131942399105356,
 'crime_and_law': 0.03402618295594812,
 'education_and_jobs': 0.036938158572321196,
 'electronics_and_hardware': 0.016033708349045165,
 'entertainment': 0.08835361235206982,
 'fashion_and_beauty': 0.007451307902511114,
 'finance_and_business': 0.06206278551627001,
 'food_and_dining': 0.02118745993746211,
 'games': 0.04599849834056857,
 'health': 0.07869924556724804,
 'history_and_geography': 0.03220994389194089,
 'home_and_hobbies': 0.025382155462877018,
 'industrial': 0.008714428090097422,
 'literature': 0.07296686896975939,
 'politics': 0.12223962603857506,
 'religion': 0.055555385841762644,
 'science_math_and_technology': 0.0854262108683093,
 'social_life': 0.04374636822484258,
 'software': 0.021607876004248874,
 'software_development': 0.044676994856554464,
 'sports_and_fitness': 0.039351999871120806,
 'transportation': 0.018158661240478786,
 'travel_and_tourism': 0.011528563106129541}

dclm_pstar = [{"domain": "adult_content", "weight": 0.001144117810399051}, {"domain": "art_and_design", "weight": 0.01249707367344882}, {"domain": "crime_and_law", "weight": 0.02923482987563418}, {"domain": "education_and_jobs", "weight": 0.042876790729263314}, {"domain": "electronics_and_hardware", "weight": 0.03452411494012139}, {"domain": "entertainment", "weight": 0.09598475243055875}, {"domain": "fashion_and_beauty", "weight": 0.00012733666316416182}, {"domain": "finance_and_business", "weight": 0.04067027113145341}, {"domain": "food_and_dining", "weight": 0.014086159108083668}, {"domain": "games", "weight": 0.06891284699600658}, {"domain": "health", "weight": 0.09942183338931967}, {"domain": "history_and_geography", "weight": 0.027098556464741227}, {"domain": "home_and_hobbies", "weight": 0.009226467887549798}, {"domain": "industrial", "weight": 0.015745768366251897}, {"domain": "literature", "weight": 0.0682512927464629}, {"domain": "politics", "weight": 0.02170366281584968}, {"domain": "religion", "weight": 0.019765203386769094}, {"domain": "science_math_and_technology", "weight": 0.2102541083699082}, {"domain": "social_life", "weight": 0.00464628155954837}, {"domain": "software", "weight": 0.04733898713026143}, {"domain": "software_development", "weight": 0.11142142430705981}, {"domain": "sports_and_fitness", "weight": 0.013072140601763493}, {"domain": "transportation", "weight": 0.009798609444870071}, {"domain": "travel_and_tourism", "weight": 0.002197370171511048}]
dclm_pstar = {f"dclm:{item['domain']}": item['weight'] for item in dclm_pstar}

dclm_pstar_pareto_diff = {'eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB v2)': -0.006765782833099365,
 'eval/downstream/mmlu_humanities_test_rc_5shot (BPB v2)': -0.017385661602020264,
 'eval/downstream/mmlu_other_test_rc_5shot (BPB v2)': 0.009604930877685436,
 'eval/downstream/mmlu_stem_test_rc_5shot (BPB v2)': 0.07365036010742188,
 'eval/downstream/winogrande_val_rc_5shot (BPB v2)': -0.025584936141967773,
 'eval/downstream/socialiqa_val_rc_5shot (BPB v2)': -0.030133485794067272,
 'eval/downstream/piqa_val_rc_5shot (BPB v2)': -0.0142478346824646,
 'eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB v2)': 0.06895178556442261,
 'eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB v2)': 0.05187523365020752,
 'eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB v2)': 0.08296090364456177,
 'eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB v2)': 0.06664419174194336,
 'eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB v2)': 0.055356621742248535,
 'eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB v2)': 0.055573225021362305,
 'eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB v2)': 0.06856685876846313,
 'eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)': 0.027955293655395508,
 'eval/downstream/hellaswag_rc_5shot (BPB v2)': -0.01006251573562622,
 'eval/downstream/csqa_val_rc_5shot (BPB v2)': 0.019213557243347168,
 'eval/downstream/codex_mbpp_gold_bpb_0shot (BPB v2)': 0.08547651767730702,
 'eval/downstream/codex_humaneval_gold_bpb_0shot (BPB v2)': 0.09800159931182861,
 'eval/downstream/arc_easy_test_rc_5shot (BPB v2)': 0.020860135555267334,
 'eval/downstream/arc_challenge_test_rc_5shot (BPB v2)': 0.029662907123565674,
 'basic_skills_arithmetic:rc::olmes': 0.1617332470913151,
 'basic_skills_coding:rc::olmes': 0.09377923040556202,
 'basic_skills_common_knowledge:rc::olmes': -0.022449946459446646,
 'basic_skills_logical_reasoning:rc::olmes': 0.023866552580687844,
 'basic_skills_string_operations:rc::olmes': 0.17098224031582898,
 'basic_skills_pattern:rc::olmes': 0.0030208730652183657,
 'mt_mbpp:bash': 0.12907993510470306,
 'mt_mbpp:c': 0.05808238366783214,
 'mt_mbpp:cpp': 0.08425639019650621,
 'mt_mbpp:csharp': 0.055141887635875464,
 'mt_mbpp:go': 0.12683272967825576,
 'mt_mbpp:haskell': 0.13547843202575582,
 'mt_mbpp:java': 0.0460935574587894,
 'mt_mbpp:javascript': 0.08682078344430177,
 'mt_mbpp:matlab': 0.18770950623611626,
 'mt_mbpp:php': 0.04329883968302617,
 'mt_mbpp:python': 0.07913380713145468,
 'mt_mbpp:r': 0.12546412515287675,
 'mt_mbpp:ruby': 0.2090636039316791,
 'mt_mbpp:rust': 0.1544673900152459,
 'mt_mbpp:scala': 0.1626278623856936,
 'mt_mbpp:swift': 0.09455429773757251,
 'mt_mbpp:typescript': 0.09594234265919976,
 'medmcqa:rc::none': 0.03849786426922619,
 'lambada': -0.015688415469722883,
 'sciq::olmo1': 0.03262081515933701,
 'squad:rc::gen2mc': -0.009724101076833247,
 'naturalqs:rc::gen2mc': -0.02105010328712953,
 'jeopardy:rc::gen2mc': -0.006787931240873402,
 'drop:rc::gen2mc': 0.00754682038641552,
 'coqa:rc::gen2mc': -0.03518194033581967,
 'ultrachat_masked_ppl': -0.002778840844915753,
 'wildchat_masked_ppl': 0.0011769254131349105}


breakpoint()


stackedu_natural = {'stack-edu:C': 0.03460056269174911,
 'stack-edu:CSharp': 0.05264555903272339,
 'stack-edu:Cpp': 0.09156360629059475,
 'stack-edu:Go': 0.010219940709777268,
 'stack-edu:Java': 0.2290669371274825,
 'stack-edu:JavaScript': 0.06493968798335928,
 'stack-edu:Markdown': 0.2112999497635124,
 'stack-edu:PHP': 0.05403765613372281,
 'stack-edu:Python': 0.1316615352877947,
 'stack-edu:Ruby': 0.010133573557099778,
 'stack-edu:Rust': 0.01036500514153347,
 'stack-edu:SQL': 0.051614847574722386,
 'stack-edu:Shell': 0.018579793390165446,
 'stack-edu:Swift': 0.01103414755816106,
 'stack-edu:TypeScript': 0.01823719775760147}

stackedu_pstar = {'stack-edu:C': 0.040545413474083136, 'stack-edu:CSharp': 0.06145614902228962, 'stack-edu:Cpp': 0.11992423990590854, 'stack-edu:Go': 0.013141668585880971, 'stack-edu:Java': 0.15971742894160593, 'stack-edu:JavaScript': 0.08711993898613768, 'stack-edu:Markdown': 0.16641522916681814, 'stack-edu:PHP': 0.060681232466575974, 'stack-edu:Python': 0.18292382056422074, 'stack-edu:Ruby': 0.01313841950835558, 'stack-edu:Rust': 0.014023586747942062, 'stack-edu:SQL': 0.018239409453020123, 'stack-edu:Shell': 0.025543226210598954, 'stack-edu:Swift': 0.014179755937669661, 'stack-edu:TypeScript': 0.02295048102889294}

stackedu_pstar_pareto_diff = {'basic_skills_coding:rc::olmes': 0.019556744532509496,
 'codex_humaneval:3shot::none': -0.0029751992996370213,
 'mbpp:3shot::none': 0.00950462534315899,
 'mt_mbpp_v2fix:bash': 0.013774151780190091,
 'mt_mbpp_v2fix:c': 0.0051739813024566095,
 'mt_mbpp_v2fix:cpp': 0.005685924728974112,
 'mt_mbpp_v2fix:csharp': 0.0022051149836528616,
 'mt_mbpp_v2fix:go': 0.00701774541883976,
 'mt_mbpp_v2fix:haskell': 0.0030367011272165456,
 'mt_mbpp_v2fix:java': 0.0024046530322927306,
 'mt_mbpp_v2fix:javascript': 0.00822884775219862,
 'mt_mbpp_v2fix:matlab': -0.0004401999903654863,
 'mt_mbpp_v2fix:php': 0.009142291717270545,
 'mt_mbpp_v2fix:python': 0.009511334668899751,
 'mt_mbpp_v2fix:r': 0.005373627704515105,
 'mt_mbpp_v2fix:ruby': 0.008480183324510726,
 'mt_mbpp_v2fix:rust': 0.006430306849438294,
 'mt_mbpp_v2fix:scala': 0.010511067523456097,
 'mt_mbpp_v2fix:swift': 0.0027267500674573175,
 'mt_mbpp_v2fix:typescript': 0.008541836046827012}




print("plotting weights...")

plot_weights(prior=dclm_natural, prediction=dclm_pstar, save_name="dclm_weights")

plot_weights(prior=stackedu_natural, prediction=stackedu_pstar, save_name="stackedu_weights")

print("plotting pareto diffs...")

plot_pareto(diff=dclm_pstar_pareto_diff, save_name="dclm_pareto_diff")
plot_pareto(diff=stackedu_pstar_pareto_diff, save_name="stackedu_pareto_diff")