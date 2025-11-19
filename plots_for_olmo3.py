# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "seaborn",
# ]
# ///


import matplotlib as mpl
from pathlib import Path
from matplotlib.font_manager import fontManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Ai2 plotting stuff from Kyle
MANROPE_BASE_PATH = Path("../FONTS").absolute()
for path in MANROPE_BASE_PATH.iterdir():
    if path.suffix == ".ttf":
        fontManager.addfont(str(path))

AI2_FONT_WEIGHTS = {
    "light": 300,
    "regular": 400,
    "medium": 500,
    "bold": 700,
}

AI2_COLORS = {
    "pink": "#f0529c",
    "teal": "#105257",
    "purple": "#b11be8",
    "green": "#0fcb8c",
    "lime": "#bef576",
    "sky": "#12cce5",
    "orange": "#f65834",
    "yellow": "#fff500",
    "error": "#fd4645",
    "warning": "#ffa31c",
    "confirmation": "#549c35",
    "information": "#2a88ef",
    "off_white": "#faf2e9",
    "dark_teal": "#0a3235",
}


def apply_ai2_theme(shade: str = "rainbow") -> None:
    """Apply the AI2 matplotlib theme to the current session."""

    # Core palette and typography.
    mpl.rcParams.update(
        {
            "font.family": "Manrope",
            "text.color": AI2_COLORS["dark_teal"],
            "axes.labelweight": AI2_FONT_WEIGHTS["medium"],
            "axes.titleweight": AI2_FONT_WEIGHTS["bold"],
            "axes.labelcolor": AI2_COLORS["dark_teal"],
            "axes.titlesize": "x-large",
            "axes.titlelocation": "left",
            "axes.edgecolor": AI2_COLORS["teal"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": "x-large",
            # "axes.prop_cycle": mpl.cycler(color=get_shade(shade)),
            "figure.titlesize": "x-large",
            "figure.titleweight": AI2_FONT_WEIGHTS["bold"],
            "figure.autolayout": False,
            "xtick.color": AI2_COLORS["teal"],
            "ytick.color": AI2_COLORS["teal"],
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "grid.color": AI2_COLORS["teal"],
            "grid.alpha": 0.15,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.grid": True,
            "axes.axisbelow": True,
            "lines.linewidth": 2.2,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "patch.edgecolor": "none",
            "patch.force_edgecolor": False,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": AI2_COLORS["teal"],
            "legend.fancybox": True,
            "legend.fontsize": "x-large",
            "legend.title_fontsize": "x-large",
        }
    )


apply_ai2_theme()

# data
dclm_natural = {
    "adult_content": 0.013552015640753739,
    "art_design": 0.014131942399105356,
    "crime_law": 0.03402618295594812,
    "education_jobs": 0.036938158572321196,
    "electronics_hardware": 0.016033708349045165,
    "entertainment": 0.08835361235206982,
    "fashion_beauty": 0.007451307902511114,
    "finance_business": 0.06206278551627001,
    "food_dining": 0.02118745993746211,
    "games": 0.04599849834056857,
    "health": 0.07869924556724804,
    "history_geography": 0.03220994389194089,
    "home_n_hobbies": 0.025382155462877018,
    "industrial": 0.008714428090097422,
    "literature": 0.07296686896975939,
    "politics": 0.12223962603857506,
    "religion": 0.055555385841762644,
    "sci_math_tech": 0.0854262108683093,
    "social_life": 0.04374636822484258,
    "software": 0.021607876004248874,
    "software_dev": 0.044676994856554464,
    "sports_fitness": 0.039351999871120806,
    "transportation": 0.018158661240478786,
    "travel": 0.011528563106129541,
}

dclm_pstar = [
    {"domain": "adult_content", "weight": 0.001144117810399051},
    {"domain": "art_design", "weight": 0.01249707367344882},
    {"domain": "crime_law", "weight": 0.02923482987563418},
    {"domain": "education_jobs", "weight": 0.042876790729263314},
    {"domain": "electronics_hardware", "weight": 0.03452411494012139},
    {"domain": "entertainment", "weight": 0.09598475243055875},
    {"domain": "fashion_beauty", "weight": 0.00012733666316416182},
    {"domain": "finance_business", "weight": 0.04067027113145341},
    {"domain": "food_dining", "weight": 0.014086159108083668},
    {"domain": "games", "weight": 0.06891284699600658},
    {"domain": "health", "weight": 0.09942183338931967},
    {"domain": "history_geography", "weight": 0.027098556464741227},
    {"domain": "home_n_hobbies", "weight": 0.009226467887549798},
    {"domain": "industrial", "weight": 0.015745768366251897},
    {"domain": "literature", "weight": 0.0682512927464629},
    {"domain": "politics", "weight": 0.02170366281584968},
    {"domain": "religion", "weight": 0.019765203386769094},
    {"domain": "sci_math_tech", "weight": 0.2102541083699082},
    {"domain": "social_life", "weight": 0.00464628155954837},
    {"domain": "software", "weight": 0.04733898713026143},
    {"domain": "software_dev", "weight": 0.11142142430705981},
    {"domain": "sports_fitness", "weight": 0.013072140601763493},
    {"domain": "transportation", "weight": 0.009798609444870071},
    {"domain": "travel", "weight": 0.002197370171511048},
]
dclm_pstar = {f"dclm:{item['domain']}": item["weight"] for item in dclm_pstar}

dclm_pstar_pareto_diff = {
    "eval/downstream/mmlu_social_sciences (BPB v2)": -0.006765782833099365,
    "eval/downstream/mmlu_humanities (BPB v2)": -0.017385661602020264,
    "eval/downstream/mmlu_other (BPB v2)": 0.009604930877685436,
    "eval/downstream/mmlu_stem (BPB v2)": 0.07365036010742188,
    "eval/downstream/winogrande (BPB v2)": -0.025584936141967773,
    "eval/downstream/socialiqa (BPB v2)": -0.030133485794067272,
    "eval/downstream/piqa (BPB v2)": -0.0142478346824646,
    "eval/downstream/minerva_math_algebra (BPB v2)": 0.06895178556442261,
    "eval/downstream/minerva_math_counting_and_probability (BPB v2)": 0.05187523365020752,
    "eval/downstream/minerva_math_geometry (BPB v2)": 0.08296090364456177,
    "eval/downstream/minerva_math_intermediate_algebra (BPB v2)": 0.06664419174194336,
    "eval/downstream/minerva_math_number_theory (BPB v2)": 0.055356621742248535,
    "eval/downstream/minerva_math_prealgebra (BPB v2)": 0.055573225021362305,
    "eval/downstream/minerva_math_precalculus (BPB v2)": 0.06856685876846313,
    "eval/downstream/gsm8k (BPB v2)": 0.027955293655395508,
    "eval/downstream/hellaswag (BPB v2)": -0.01006251573562622,
    "eval/downstream/csqa (BPB v2)": 0.019213557243347168,
    "eval/downstream/mbpp (BPB v2)": 0.08547651767730702,
    "eval/downstream/humaneval (BPB v2)": 0.09800159931182861,
    "eval/downstream/arc_easy (BPB v2)": 0.020860135555267334,
    "eval/downstream/arc_challenge (BPB v2)": 0.029662907123565674,
    "basic_skills_arithmetic": 0.1617332470913151,
    "basic_skills_coding": 0.09377923040556202,
    "basic_skills_common_knowledge": -0.022449946459446646,
    "basic_skills_logical_reasoning": 0.023866552580687844,
    "basic_skills_string_operations": 0.17098224031582898,
    "basic_skills_pattern": 0.0030208730652183657,
    "mt_mbpp:bash": 0.12907993510470306,
    "mt_mbpp:c": 0.05808238366783214,
    "mt_mbpp:cpp": 0.08425639019650621,
    "mt_mbpp:csharp": 0.055141887635875464,
    "mt_mbpp:go": 0.12683272967825576,
    "mt_mbpp:haskell": 0.13547843202575582,
    "mt_mbpp:java": 0.0460935574587894,
    "mt_mbpp:javascript": 0.08682078344430177,
    "mt_mbpp:matlab": 0.18770950623611626,
    "mt_mbpp:php": 0.04329883968302617,
    "mt_mbpp:python": 0.07913380713145468,
    "mt_mbpp:r": 0.12546412515287675,
    "mt_mbpp:ruby": 0.2090636039316791,
    "mt_mbpp:rust": 0.1544673900152459,
    "mt_mbpp:scala": 0.1626278623856936,
    "mt_mbpp:swift": 0.09455429773757251,
    "mt_mbpp:typescript": 0.09594234265919976,
    "medmcqa": 0.03849786426922619,
    "lambada": -0.015688415469722883,
    "sciq": 0.03262081515933701,
    "squad": -0.009724101076833247,
    "naturalqs": -0.02105010328712953,
    "jeopardy": -0.006787931240873402,
    "drop": 0.00754682038641552,
    "coqa": -0.03518194033581967,
    "ultrachat": -0.002778840844915753,
    "wildchat": 0.0011769254131349105,
}

stackedu_natural = {
    "stack-edu:C": 0.03460056269174911,
    "stack-edu:CSharp": 0.05264555903272339,
    "stack-edu:Cpp": 0.09156360629059475,
    "stack-edu:Go": 0.010219940709777268,
    "stack-edu:Java": 0.2290669371274825,
    "stack-edu:JavaScript": 0.06493968798335928,
    "stack-edu:Markdown": 0.2112999497635124,
    "stack-edu:PHP": 0.05403765613372281,
    "stack-edu:Python": 0.1316615352877947,
    "stack-edu:Ruby": 0.010133573557099778,
    "stack-edu:Rust": 0.01036500514153347,
    "stack-edu:SQL": 0.051614847574722386,
    "stack-edu:Shell": 0.018579793390165446,
    "stack-edu:Swift": 0.01103414755816106,
    "stack-edu:TypeScript": 0.01823719775760147,
}

stackedu_pstar = {
    "stack-edu:C": 0.040545413474083136,
    "stack-edu:CSharp": 0.06145614902228962,
    "stack-edu:Cpp": 0.11992423990590854,
    "stack-edu:Go": 0.013141668585880971,
    "stack-edu:Java": 0.15971742894160593,
    "stack-edu:JavaScript": 0.08711993898613768,
    "stack-edu:Markdown": 0.16641522916681814,
    "stack-edu:PHP": 0.060681232466575974,
    "stack-edu:Python": 0.18292382056422074,
    "stack-edu:Ruby": 0.01313841950835558,
    "stack-edu:Rust": 0.014023586747942062,
    "stack-edu:SQL": 0.018239409453020123,
    "stack-edu:Shell": 0.025543226210598954,
    "stack-edu:Swift": 0.014179755937669661,
    "stack-edu:TypeScript": 0.02295048102889294,
}

stackedu_pstar_pareto_diff = {
    "basic_skills_coding": 0.019556744532509496,
    "humaneval": -0.0029751992996370213,
    "mbpp": 0.00950462534315899,
    "mt_mbpp:bash": 0.013774151780190091,
    "mt_mbpp:c": 0.0051739813024566095,
    "mt_mbpp:cpp": 0.005685924728974112,
    "mt_mbpp:csharp": 0.0022051149836528616,
    "mt_mbpp:go": 0.00701774541883976,
    "mt_mbpp:haskell": 0.0030367011272165456,
    "mt_mbpp:java": 0.0024046530322927306,
    "mt_mbpp:javascript": 0.00822884775219862,
    "mt_mbpp:matlab": -0.0004401999903654863,
    "mt_mbpp:php": 0.009142291717270545,
    "mt_mbpp:python": 0.009511334668899751,
    "mt_mbpp:r": 0.005373627704515105,
    "mt_mbpp:ruby": 0.008480183324510726,
    "mt_mbpp:rust": 0.006430306849438294,
    "mt_mbpp:scala": 0.010511067523456097,
    "mt_mbpp:swift": 0.0027267500674573175,
    "mt_mbpp:typescript": 0.008541836046827012,
}


def plot_weights(prior: dict[str, float], prediction: dict[str, float], save_name: str):
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

    _, ax = plt.subplots(figsize=(8, 6), layout="compressed")
    ax.ticklabel_format(useMathText=True)
    ax.tick_params(axis="x", labelrotation=90)

    pallette = {
        "Natural": "#105257",
        "Ours": "#F0529C",
    }

    df_sorted = df[df["type"] == "Natural"].sort_values(by="value", ascending=False)
    df["variable"] = pd.Categorical(df["variable"], categories=df_sorted["variable"], ordered=True)
    df["variable"] = [label.replace("dclm:", "").replace("stack-edu:", "") for label in df["variable"]]
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pallette, ax=ax)

    ax.legend(
        frameon=False,
        handlelength=0.4,
        ncol=2,
    )

    ax.set_xlabel("Domain")
    ax.set_ylabel("Weight")

    plt.savefig(
        f"{save_name}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_pareto(diff: dict[str, float], save_name: str, group=False):
    diff_dict = {k.split("/")[-1].split(" ")[0].split("::")[0]: v for k, v in diff.items()}
    plt.figure(figsize=(8, 6))
    diff = pd.Series(diff_dict)
    colors = [AI2_COLORS["confirmation"] if val > 0 else AI2_COLORS["error"] for val in diff]
    ax = diff.plot(kind="bar", color=colors)
    ax.grid(axis="x", visible=False)

    if group:
        mapping = {}
        for k in diff_dict:
            if k.startswith("mmlu"):
                mapping[k] = "mmlu"
            elif k.startswith("minerva_math"):
                mapping[k] = "minerva_math"
            elif k.startswith("minerva_math"):
                mapping[k] = "minerva_math"
            elif k.startswith("mt_mbpp"):
                mapping[k] = "mt_mbpp"
            elif k.startswith("basic_skills"):
                mapping[k] = "basic_skills"
            elif k.startswith("arc"):
                mapping[k] = "arc"
            else:
                mapping[k] = k

        # Optionally hide original long labels
        ax.set_xticklabels([""] * len(diff.index), rotation=90)

        # 3. Collect indices for each group
        groups = {}
        for i, key in enumerate(diff.index):
            grp = mapping[key]
            groups.setdefault(grp, []).append(i)

        # 4. Draw brackets + labels under x-axis
        trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
        y_bracket = -0.05  # vertical position below axis
        h_bracket = 0.01  # bracket height

        for grp, idx_list in groups.items():
            xmin = min(idx_list) - 0.2  # extend to bar edges (width ~ 0.8)
            xmax = max(idx_list) + 0.2
            xmid = 0.5 * (xmin + xmax)

            # horizontal line
            ax.plot(
                [xmin, xmax],
                [y_bracket, y_bracket],
                transform=trans,
                linewidth=1,
                clip_on=False,
                color=AI2_COLORS["teal"],
            )

            # two vertical ticks of the bracket
            ax.plot(
                [xmin, xmin],
                [y_bracket, y_bracket + h_bracket],
                transform=trans,
                linewidth=1,
                clip_on=False,
                color=AI2_COLORS["teal"],
            )
            ax.plot(
                [xmax, xmax],
                [y_bracket, y_bracket + h_bracket],
                transform=trans,
                linewidth=1,
                clip_on=False,
                color=AI2_COLORS["teal"],
            )

            # group label centered under bracket
            ax.text(
                xmid,
                y_bracket - 0.02,
                grp,
                ha="center",
                va="top",
                rotation=90,
                transform=trans,
                color=AI2_COLORS["teal"],
            )

        # 5. Give some extra bottom margin so labels are visible
        plt.subplots_adjust(bottom=0.25)

    plt.ylabel("BPB Improvement (↑)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        f"{save_name}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


print("plotting weights...")

plot_weights(prior=dclm_natural, prediction=dclm_pstar, save_name="dclm_weights")
plot_weights(prior=stackedu_natural, prediction=stackedu_pstar, save_name="stackedu_weights")

print("plotting pareto diffs...")

plot_pareto(diff=dclm_pstar_pareto_diff, save_name="dclm_pareto_diff", group=True)
plot_pareto(diff=stackedu_pstar_pareto_diff, save_name="stackedu_pareto_diff")
