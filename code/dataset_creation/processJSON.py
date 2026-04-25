import json
import os
import re
from pathlib import Path

# ---- Configuration ----
input_json_file = "proposals/source/06.json"
output_folder = "proposals/abstract_txt"
allowed_call_tokens = ("STG", "COG", "ADV")
# -----------------------


def sanitize_filename(name: str) -> str:
    if not name:
        return "missing_grantDoi"

    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1F]', "_", name)
    name = name.strip().rstrip(".")
    return name or "missing_grantDoi"


def extract_call_identifiers(project: dict) -> list[str]:
    calls = (
        project.get("relations", {})
        .get("associations", {})
        .get("call", [])
    )

    if isinstance(calls, dict):
        calls = [calls]

    identifiers = []
    for call in calls:
        identifier = call.get("identifier")
        if identifier:
            identifiers.append(identifier)
    return identifiers


def is_allowed_call(project: dict) -> bool:
    identifiers = extract_call_identifiers(project)
    for identifier in identifiers:
        upper_id = identifier.upper()
        if any(token in upper_id for token in allowed_call_tokens):
            return True
    return False


def main():
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    hits = data.get("hits", {}).get("hit", data.get("result", {}).get("hits", {}).get("hit", []))
    if isinstance(hits, dict):
        hits = [hits]

    created = 0
    skipped_missing = 0
    skipped_call = 0

    for i, hit in enumerate(hits, start=1):
        project = hit.get("project", {})

        if not is_allowed_call(project):
            skipped_call += 1
            print(f"Skipping hit {i}: call identifier does not contain STG, COG, or ADV")
            continue

        objective = project.get("objective")
        grant_doi = project.get("identifiers", {}).get("grantDoi")

        if not objective or not grant_doi:
            skipped_missing += 1
            print(f"Skipping hit {i}: missing objective or grantDoi")
            continue

        filename = sanitize_filename(grant_doi) + ".txt"
        output_path = os.path.join(output_folder, filename)

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(objective)

        created += 1
        print(f"Created: {output_path}")

    print(
        f"\nDone. Created {created} files, "
        f"skipped {skipped_call} for call filter, "
        f"skipped {skipped_missing} for missing data."
    )


if __name__ == "__main__":
    main()