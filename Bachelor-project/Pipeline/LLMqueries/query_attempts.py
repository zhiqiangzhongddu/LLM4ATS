predictor_instruction = {
    "role": "system",
    "content": "You are an expert in biomedicine and chemistry, specializing in molecules."
}

system_instruction = {
    "role": "system",
    "content": "You are an expert in biomedicine and chemistry, specializing in molecules. "
    "You value consistency and accuracy in your responses. "
    "You do not give an anwser if you are not confident in the response you provide. "
}
system_instruction_2 = {
    "role": "system",
    "content": "You are an expert in biomedicine and chemistry, specializing in molecules. "
    "You value consistency and accuracy in your responses. "
    "You do not give an anwser if you are not confident in the response you provide. "
    "You are also aware that the user values consistency in the responses you provide. "
}

system_instruction_3 = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
}





query_attempts = {
    "default": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that have the highest correlation with the {} in a any given molecule? Please select the properties that are the most important. "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "{}\n"
        }
    ],
    "form_1":[
        system_instruction,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties, ONLY the given list properties, that have the highest correlation with the {} in a any given molecule? Please select the properties that are the most important. "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "{}\n"
        }
    ],
    "form_2":[
        system_instruction,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that have the highest correlation with the {} in a any given molecule? Please select the properties that are the most important. "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "When giving your response please ensure that you give both the name and the description of the property. "
            "{}\n"
        }
    ],
    "form_3":[
        system_instruction_2,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that have the highest correlation with the {} in a any given molecule? Please select the properties that are the most important. "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "When giving your response please ensure that you give both the name and the description of the property. "
            "{}\n"
        }
    ],
        "form_4":[ # descriptor, max_props, target_task, target task, max_props
        system_instruction_2,
        {
            "role": "user",
            "content": "Here is a list of molecular descriptors:\n<descriptors>\n{}</descriptors>\n\n"
            "Please carefully review this list of descriptors and select an absolut maximum of {} descriptors that you think are most likely to have a correlation with the {} of a molecule. "
            "Focus on selecting the descriptors with the highest potential correlation to {} of a molecule.\nDo not provide any explanations for your selections. Just list out the descriptors you have chosen.\n"
            "Please format your output with the selected descriptors listed one per line inside <selected_descriptors> tags, like this:\n\n<selected_descriptors>\n- Descriptor 1\n- Descriptor 2\n...\n- Descriptor {}\n</selected_descriptors>"
        }
    ],
    "form_5":[ # descriptor, max_props, target_task, target task, max_props
        system_instruction_2,
        {
            "role": "user",
            "content": "Here is a list of molecular descriptors:\n<descriptors>\n{}</descriptors>\n\n"
            "Please carefully review this list of descriptors and select an absolut maximum of {} descriptors that you think are most likely to have a correlation with the {} of a molecule. "
            "Focus on selecting the descriptors with the highest potential correlation to {} of a molecule.\nDo not provide any explanations for your selections. Just list out the descriptors you have chosen.\n"
            "Please format your output with the selected descriptors listed one per line inside <selected_descriptors> tags, like this:\n\n<selected_descriptors>\n- Descriptor 1\n- Descriptor 2\n...\n- Descriptor {}\n</selected_descriptors>\n"
            "Do not deviate from this anwser format and return both the whole name and the given description."
        }
    ],
    "g_form_1":[
        system_instruction,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that could potentially have any correlation with the {} in a any given molecule? "
            "Please select the properties that are the most important if there are more than the maximum amount. "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "{}\n"
        }
    ],
    "g_form_2":[
        system_instruction_3,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that could potentially have any correlation with the {} in a any given molecule? "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "{}\n"
        }
    ],
    "g_form_2":[
        system_instruction_3,
        {
            "role": "user",
            "content": "Can you provide a subset of maximum size of {} out of the given list of properties that *could* potentially have correlation with the {} in a any given molecule? "
            "If you are confident in the fact that no property does not any correlation with the target task, then please explicitly and only state that by writing \"NO ANWSER\". "
            "The properties are given in the following format: <name>: <description>. If there is no description, then the name should be self-explanatory. "
            "{}\n"
        }
    ],
            "g_form_4":[ # descriptor, aux_tasks, target_task
        system_instruction_2,
        {
            "role": "user",
            "content": "Here is a list of molecular descriptors:\n<descriptors>\n{}</descriptors>\n\n"
            "Please carefully review this list of descriptors and select an absolut maximum of {} descriptors that you think are most likely to have a correlation with the {} of a molecule. Focus on selecting the descriptors with the highest potential correlation to toxicity.\nDo not provide any explanations for your selections. Just list out the descriptors you have chosen.\n"
            "Please format your output with the selected descriptors listed one per line inside <selected_descriptors> tags, like this:\n\n<selected_descriptors>\nDescriptor 1\nDescriptor 2\n...\nDescriptor {}\n</selected_descriptors>\n do not deviate from this answer format and return both the name and the given description."
        }
    ],
}