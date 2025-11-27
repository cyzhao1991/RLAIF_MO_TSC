import copy

clip_env_prompts = {
    "metaworld_sweep-into-v2": "The green cube is in the hole.", # unsolved there is reward issue
    "metaworld_drawer-open-v2": "The drawer is opened.", # let's try the flipped version.
    "metaworld_door-open-v2": "The safe door is opened.", # let's try the flipped version.
    "metaworld_soccer-v2": "The soccer ball is in the goal.", # not solved, there is reward issue

    "CartPole-v1": "pole vertically upright on top of the cart.",
    "CartPole-v2": "pole vertically upright on top of the cart.",
    "CartPoleContinous-v1": "pole vertically upright on top of the cart.",

    "Parking-v0": "park the vehicle at the target green slot.",
    
    "softgym_RopeFlattenEasy": "The blue rope is straightened.",
    "softgym_PassWater": "The container, which holds water, is as close to the red circle as possible without causing too many water droplets to spill.",
    "softgym_ClothFoldDiagonal": "The cloth is folded diagonally from top left corner to bottom right corner.",

    "metaworld_hammer-v2": "The screw is hammered into the wall.",
    "tsc_single-intersection": None,
}

# what RL-VLM-F uses
goal_env_prompts = {
    "metaworld_sweep-into-v2": "to minimize the distance between the green cube and the hole", # unsolved there is reward issue
    "metaworld_drawer-open-v2": "to open the drawer", # let's try the flipped version.
    "metaworld_door-open-v2": "to open the safe door", # let's try the flipped version.
    "metaworld_soccer-v2": "to move the soccer ball into the goal", # not solved, there is reward issue
    "CartPole-v1": "to balance the brown pole on the black cart to be upright",
    "CartPole-v2": "pole vertically upright on top of the cart.",
    "CartPoleContinous-v1": "to balance the brown pole on the black cart to be upright",
    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",

    "metaworld_hammer-v2": "to hammer the screw into the wall",
    "tsc_single-intersection": "to ensure optimized traffic flow with high throughput and low delay",
    "Parking-v0": "to park the yellow vehicle at the target green slot.",
}

prefix_env_prompts = {
    "metaworld_sweep-into-v2": "", # unsolved there is reward issue
    "metaworld_drawer-open-v2": "", # let's try the flipped version.
    "metaworld_door-open-v2": "", # let's try the flipped version.
    "metaworld_soccer-v2": "", # not solved, there is reward issue
    "CartPole-v1": "to balance the brown pole on the black cart to be upright",
    "CartPole-v2": "pole vertically upright on top of the cart.",
    "CartPoleContinous-v1": "to balance the brown pole on the black cart to be upright",
    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",

    "metaworld_hammer-v2": "to hammer the screw into the wall",
    "tsc_single-intersection": "You will be given two scenes of traffic junction with traffic stats. Please help with analyze and compare the two scenes.",
    "Parking-v0": "to park the yellow vehicle at the target green slot.",
}



##########################################################################
### asking gemini to output a preference with 2 stage analysis ###############
#########################################################################
gemini_free_query_prompt1 =  """
Consider the following two images:
Image 1:
"""

gemini_free_query_prompt2 = """
Image 2:
"""

gemini_free_query_env_prompts = {}
gemini_free_query_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

for env_name, prompt in goal_env_prompts.items():
    gemini_free_query_env_prompts[env_name] = gemini_free_query_template.format(prompt)
    
### preference summary prompt
gemini_summary_env_prompts = {}

### template 1 
gemini_summary_template = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
""" 

for env_name, prompt in goal_env_prompts.items():
    gemini_summary_env_prompts[env_name] = gemini_summary_template.format(prompt, "{}")
    
######################################################################
### asking gemini to output a preference with 1 stage analysis ###############
######################################################################

gemini_single_query_prompt_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
"""

gemini_single_query_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_single_query_env_prompts[env_name] = gemini_single_query_prompt_template.format(prompt)
    
######################################################################
### asking gemini to output a preference with only labeling ###############
######################################################################
gemini_single_query_no_analysis_prompt_template = """
The goal is {}. Is the goal better achieved in Image 1 or Image 2?
At the end of the response, reply a single line of: 
0 if the goal is better achieved in Image 1, 
1 if it is better achieved in Image 2, or
-1 if there is no difference or if it is unclear.
"""

### asking gemini to output score
gemini_score_prompt_start = """
Consider the following image:
"""

gemini_score_template = """
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
"""

gemini_score_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_env_prompts[env_name] = gemini_score_template.format(prompt)

gemini_score_summary_template = """
Based on the text below to the questions: 
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
{}

Please reply a single line of the score the text has given.
Reply -1 if the text is unsure.
"""

### text related prompts
gpt_free_text_query_env_prompts = """
1. What is shown in scene 1?
2. What is shown in scene 2?
3. The goal is {}. Is there any difference between scene 1 and scene 2 in terms of achieving the goal?
"""

gpt_summary_text_query_env_prompts = """
Based on the text below to the questions:
1. What is shown in scene 1?
2. What is shown in scene 2?
3. The goal is {}. Is there any difference between scene 1 and scene 2 in terms of achieving the goal?
{}

Is the goal better achieved in scene 1 or scene 2?
Reply a single line of 0 if the goal is better achieved in scene 1, or 1 if it is better achieved in scene 2.
Reply -1 if the text is unsure or there is no difference.
"""

gemini_score_summary_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_summary_env_prompts[env_name] = gemini_score_summary_template.format(prompt, "{}")
    
    
### GPT uses the same prompt template as gemini
gpt_free_query_env_prompts = {}
# gpt_free_query_template = copy.deepcopy(gemini_free_query_template)
gpt_free_query_template = copy.deepcopy(gpt_free_text_query_env_prompts)
for env_name, prompt in goal_env_prompts.items():
    gpt_free_query_env_prompts[env_name] = gpt_free_query_template.format(prompt)

gpt_summary_env_prompts = {}
# gpt_summary_template = copy.deepcopy(gemini_summary_template)
gpt_summary_template = copy.deepcopy(gpt_summary_text_query_env_prompts)
for env_name, prompt in goal_env_prompts.items():
    gpt_summary_env_prompts[env_name] = gpt_summary_template.format(prompt, "{}")

gpt_prefix_env_prompts = prefix_env_prompts

gpt_score_query_env_prompts = {}
gpt_score_template = copy.deepcopy(gemini_score_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_query_env_prompts[env_name] = gpt_score_template.format(prompt)

gpt_score_summary_env_prompts = {}
gpt_score_summary_template = copy.deepcopy(gemini_score_summary_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_summary_env_prompts[env_name] = gpt_score_summary_template.format(prompt, "{}")