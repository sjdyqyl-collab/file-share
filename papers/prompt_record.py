import agenta as ag
import os

os.environ["AGENTA_API_KEY"] = "zGPS3SLX.142f73b0243038604ecb8e42a09447d2afb5b995701d1b81620506ac7cb9bd33"

ag.init()

config = ag.ConfigManager.get_from_registry(
    app_slug = "paper_to_real",
    variant_slug = "fire",
    variant_version = 2,
)

prompt_template = config["prompt"]["messages"][0]["content"]
#prompt_template.format(paper_path="/home/wzc/papers/EP/paper.md",save_path="/home/wzc/papers/EP/concise_paper.md")
with open('example.py','w') as f:
    f.write(prompt_template)
