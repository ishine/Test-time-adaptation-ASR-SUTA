"""
A dedicated helper to manage templates and prompt building.
"""
import json
import os.path as osp


SRC_DIR = "prompts"


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str, verbose: bool = False):
        self._verbose = verbose
        file_name = osp.join(SRC_DIR, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(self, input) -> str:
        res = self.template["prompt"].format(**input)
        if self._verbose:
            print(res)
        return res
