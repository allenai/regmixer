import click
import json
from enum import Enum


class SourceCategory(Enum):
    ADULT_CONTENT = "adult_content"
    ART_AND_DESIGN = "art_and_design"
    SOFTWARE_DEVELOPMENT = "software_development"
    EDUCATION_AND_JOBS = "education_and_jobs"
    ELECTRONICS_AND_HARDWARE = "electronics_and_hardware"
    ENTERTAINMENT = "entertainment"
    CRIME_AND_LAW = "crime_and_law"
    SOCIAL_LIFE = "social_life"
    FASHION_AND_BEAUTY = "fashion_and_beauty"
    FINANCE_AND_BUSINESS = "finance_and_business"
    FOOD_AND_DINING = "food_and_dining"
    GAMES = "games"
    HEALTH = "health"
    HISTORY_AND_GEOGRAPHY = "history_and_geography"
    HOME_AND_HOBBIES = "home_and_hobbies"
    INDUSTRIAL = "industrial"
    LITERATURE = "literature"
    POLITICS = "politics"
    RELIGION = "religion"
    SCIENCE_MATH_AND_TECHNOLOGY = "science_math_and_technology"
    SOFTWARE = "software"
    SPORTS_AND_FITNESS = "sports_and_fitness"
    TRANSPORTATION = "transportation"
    TRAVEL_AND_TOURISM = "travel_and_tourism"


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--author", type=str)
@click.option("--name-contains", type=str)
def reformat_runs(author, input_file, name_contains, output_file):
    fields = ["name", "created_at"]
    with open(input_file, "r") as input_file, open(output_file, "w", newline="") as output_file:
        data = json.load(input_file)

        output = []
        for group in data.values():
            for run in group:
                run_author = run.get("author", "")

                if author and run_author != author:
                    continue

                if name_contains and name_contains not in run["name"]:
                    continue

                subset = {field: run.get(field, "") for field in fields}

                group_id = (
                    run.get("config", {})
                    .get("trainer", {})
                    .get("callbacks", {})
                    .get("wandb", {})
                    .get("group", "")
                )
                sources = run.get("config", {}).get("dataset", {}).get("source_mixture_config", {})
                max_tokens = sources.get("max_tokens", 0)
                mixture = sources.get("source_configs", [])

                evals = run.get("summary", {})
                evals = {k: v for k, v in run.get("summary", {}).items() if k.startswith("eval/")}
                subset["sources"] = [
                    {
                        "name": source.get("source_name"),
                        "ratio": float(source.get("target_ratio", 0.0)),
                        "max_repetitions": source.get("max_repetition_ratio", 1.0),
                    }
                    for source in mixture
                ]

                # For clarity add all the other source categories with 0.0 ratio
                for source_name in SourceCategory:
                    if source_name.value not in [mix["source_name"] for mix in mixture]:
                        subset["sources"].append(
                            {
                                "name": source_name.value,
                                "ratio": 0.0,
                                "max_repetitions": 1.0,
                            }
                        )

                subset["sources"].sort(key=lambda x: x["name"])
                subset["author"] = author
                subset["max_tokens"] = max_tokens
                subset["train_steps"] = run.get("summary", {}).get("_step", 0)
                subset["runtime_sec"] = run.get("summary", {}).get("_runtime", 0.0)
                subset["evals"] = evals
                subset["group_id"] = group_id
                subset["run_id"] = run["id"]
                output.append(subset)

        print(f"Total runs reported: {len(output)}")
        json.dump(output, output_file, indent=4)


if __name__ == "__main__":
    reformat_runs()
