# get the description according to class label
import json

class ObjectDescriptionLoader:
    def __init__(self, jsonl_path):
        """
        Loads a JSONL file into memory as a dictionary for quick lookups.

        Args:
            jsonl_path (str): Path to the JSONL file.
        """
        self.objects = self.load_jsonl(jsonl_path)

    def load_jsonl(self, jsonl_path):
        """
        Reads a JSONL file and stores object descriptions in a dictionary.

        Args:
            jsonl_path (str): Path to the JSONL file.

        Returns:
            dict: A dictionary mapping class labels to their details.
        """
        objects = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                class_label = obj.get("name")  # Use "name" as the class label
                if class_label:
                    objects[class_label] = obj  # Store full object info
        return objects

    def get_description(self, class_label):
        """
        Retrieves the full object description for a given class label.

        Args:
            class_label (str): The class label to look up.

        Returns:
            dict or str: The object's details if found, else "Class label not found".
        """
        return self.objects.get(class_label, "Class label not found")

# Example Usage
jsonl_path = "./descriptions.jsonl"  # Replace with your actual JSONL file path
loader = ObjectDescriptionLoader(jsonl_path)

# Query an object by its class label (name)
class_label = "001_a_and_w_root_beer_soda_pop_bottle"
object_info = loader.get_description(class_label)
print(object_info)  # Output: Full object details
