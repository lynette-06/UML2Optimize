import json
from pathlib import Path


def parse_model(file_path):
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        model = json.load(file)

    if "components" not in model:
        raise ValueError("The model must contain components.")

    if not isinstance(model["components"], list):
        raise ValueError("Components must be a list.")

    for component in model["components"]:
        if "name" not in component:
            raise ValueError("Every component must have a name.")

        if "attributes" not in component:
            raise ValueError(
                f"Component {component['name']} has no attributes."
            )

    return model