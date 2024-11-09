def create_injection_prompt(request, prompt_template, split_token):
    injection_prompt = prompt_template.format(
        h=request["subject"], 
        r=request["relation"], 
        t=request["target_new"]["str"], 
        split_token = split_token,
        prompt=request["prompt"].format(request["subject"])
    )
    return injection_prompt