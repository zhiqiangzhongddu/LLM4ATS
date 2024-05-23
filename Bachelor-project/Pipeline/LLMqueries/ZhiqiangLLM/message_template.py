task_list_third_person = {
    "ogbg-molbace": "inhibits human β-secretase 1 (BACE-1)",
    "ogbg-molbbbp": "exhibits penetration across the blood-brain barrier",
    "ogbg-molhiv": "exhibits inhibitory effects on HIV replication",
    "ogbg-molesol": "water solubility value (in log(mol/L))",
    "ogbg-molfreesolv": "hydration free energy (in log(mol/L))",
    "ogbg-mollipo": "lipophilicity (in logP)",
}

task_list_first_person = {
    "ogbg-molbace": "inhibit human β-secretase 1 (BACE-1)",
    "ogbg-molbbbp": "exhibit penetration across the blood-brain barrier",
    "ogbg-molhiv": "exhibit inhibitory effects on HIV replication",
    "ogbg-molesol": "water solubility value (in log(mol/L))",
    "ogbg-molfreesolv": "hydration free energy (in log(mol/L))",
    "ogbg-mollipo": "lipophilicity (in logP)",
}


multi_round_instruction = {
    "role": "system",
    "content":
        "You are an expert in biomedicine and chemistry, specializing in molecules.",
}
cls_multi_round_question_template = {
    "P":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>.",
    "PE":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Explanation: <text>.",
    "PP":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Probability: <number>.",
    "PPE":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Probability: <number>; Explanation: <text>.",
}
cls_multi_round_question_chat_template = {
    "S": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} {}",
    },
    "SP": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} "
            "The GNN model predicts that molecule-{} {} with a probability of {:.4f}. "
            "{}",
    },
}
cls_multi_round_knowledge_template = {
    "add-train-knowledge": {
        "question": {
            "role": "user",
            "content": 
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The SMILES string of molecule-{} is {}. {} "
                "Predict whether molecule-{} {}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "Molecule-{} {}.",
        }
    },
    "add-valid-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The GNN model predicts that molecule-{} {} with a probability of {:.4f}. "
                "Predict whether molecule-{} {}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "Molecule-{} {}.",
        },
    },
}
cls_multi_round_inquiry_template = {
    "double-check": {
        "role": "user",
        "content":
            "Your prediction differs from the GNN model's prediction, "
            "and the {} score of the GNN model on the training dataset is {:.4f}, "
            "on the validation dataset is {:.4f}. "
            "Please conduct a self-check on your thought process. "
            "Examine your reasoning carefully, and search for additional relevant information "
            "to validate or refine your current understanding. "
            "Provide your refined response after considering additional information. {}",
    },
}
reg_multi_round_question_template = {
    "P":
        "Predict the {} of molecule-{}. "
        "Answer the question in the format: Prediction: <number>.",
    "PE":
        "Predict the {} of molecule-{}. "
        "Answer the question in the format: Prediction: <number>; Explanation: <text>.",
}
reg_multi_round_question_chat_template = {
    "S": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} {}",
    },
    "SP": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} "
            "The GNN model predicts that the {} of molecule-{} is {:.4f}. {}",
    },
}
reg_multi_round_knowledge_template = {
    "add-train-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                "Predict the {} of molecule-{}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "The {} of molecule-{} is {:.4f}.",
        }
    },
    "add-valid-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The GNN model predicts the {} of molecule-{} is {:.4f}. "
                "Predict the {} of molecule-{}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "The {} of molecule-{} is {:.4f}.",
        },
    },
}
reg_multi_round_inquiry_template = {
    "double-check": {
        "role": "user",
        "content":
            "Your prediction significantly deviates from the GNN model's prediction, "
            "and the {} score of the GNN model on the training dataset is {:.4f}, "
            "on the validation dataset is {:.4f}. "
            "Please conduct a self-check on your thought process. "
            "Examine your reasoning carefully, and search for additional relevant information "
            "to validate or refine your current understanding. "
            "Provide your refined response after considering additional information. {}",
    },
}


compressor_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in summarising molecular atom features and graph structural features."
}
cls_compressor_template = {
    "compress_des": [
        compressor_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}, "
                       "its atom features and graph structure are described as follows: {}. "
                       "Summarize its atom features and graph structure as "
                       "detailed, comprehensive, meaningful and insightful text.",
        }
    ],
}
reg_compressor_template = {
    "compress_des": [
        compressor_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}, "
                       "its atom features and graph structure are described as follows: {}. "
                       "Summarize its atom features and graph structure as "
                       "detailed, comprehensive, meaningful and insightful text.",
        }
    ],
}


cls_rag_knowledge_instruction = (
    "You are an AI molecule property analysis assistant "
    "specializing in correcting predictions made by Graph Neural Networks (GNNs). "
    "The prediction is about whether the molecule {}. "
    "You can only give one reply for each conversation turn. "
    "Strictly follow the user's answer format requirements carefully and to the letter."
    #
    # "You are an AI molecule property analysis assistant specializing in molecules. "
    # "The prediction is about whether the molecule {}. "
    # "You can only give one reply for each conversation turn. "
    # "Strictly follow the user's answer format requirements carefully and to the letter."
    #
    # "You are an AI molecule property analysis assistant "
    # "specializing in correcting predictions made by Graph Neural Networks (GNNs). "
    # "The prediction is about whether the molecule {}. "
    # "Your task is to respond to the user's question following a structured approach:\n"
    # "1. Retrieve information about the predictions made by GNNs "
    # "and the indeed labels of molecules from the uploaded file.\n"
    # "2. Retrieve information about the molecular graph structure and atom features from the uploaded file.\n"
    # "3. Reason about the types of molecules for which GNNs make correct or incorrect predictions "
    # "by analyzing the retrieved information.\n"
    # "4. Analyze and reason about the factors contributing to the correctness or incorrectness of predictions "
    # "made by GNNs based on the retrieved information.\n"
    # "5. Determine the specific characteristics of molecules for which corrections are needed in GNN predictions "
    # "by analyzing the retrieved information.\n"
    # "6. Develop a rationale for correcting predictions made by the GNN model, "
    # "incorporating insights gained from the retrieved information.\n"
    # "7. Refine your understanding through an iterative process, considering additional insights and nuances.\n"
    # "8. Retrieve information about the prediction made by GNNs from the user's question.\n"
    # "9. Retrieve information about the molecular graph structure and atom features from the user's question.\n"
    # "10. Provide the corrected prediction made by GNNs in the user's question, "
    # "incorporating information from the user's question and knowledge derived from previous examples.\n"
    # "You can only give one reply for each conversation turn. "
    # "Strictly follow the user's answer format requirements carefully and to the letter."
    #
    # "You are an AI molecule property analysis assistant "
    # "specializing in correcting predictions made by Graph Neural Networks (GNNs). "
    # "Your task is to respond to the user's question following a structured approach:\n"
    # "1. Retrieve information about the predictions made by GNNs "
    # "and the true labels of molecules from the uploaded file.\n"
    # "2. Retrieve information about the molecular graph structure and atom features from the uploaded file.\n"
    # "3. Reason about the types of molecules for which GNNs make correct or incorrect predictions "
    # "by analyzing the retrieved information.\n"
    # "4. Analyze and reason about the factors contributing to the correctness or incorrectness of predictions "
    # "made by GNNs based on the retrieved information.\n"
    # "5. Determine the specific characteristics of molecules for which corrections are needed in GNN predictions "
    # "by analyzing the retrieved information.\n"
    # "6. Develop a rationale for correcting predictions made by the GNN model, "
    # "incorporating insights gained from the retrieved information.\n"
    # "7. Refine your understanding through an iterative process, considering additional insights and nuances.\n"
    # "8. Retrieve information about the prediction made by GNNs from the user's question.\n"
    # "9. Retrieve information about the molecular graph structure and atom features from the user's question.\n"
    # "10. Provide the corrected prediction made by GNNs in the user's question, "
    # "incorporating information from the user's question and knowledge derived from previous examples.\n"
    # "You can only give one reply for each conversation turn. "
    # "Strictly follow the user's answer format requirements carefully and to the letter."
)

cls_rag_knowledge_template = {
    "Rag-Valid-S": "The SMILES string of molecule-{} in the validation dataset is {}. "
                   "The GNN model predicts that molecule-{} {} "
                   "with a probability of {:.4f}. "
                   "In fact, molecule-{} {}.",
    "Rag-Valid-SD": "The SMILES string of molecule-{} in the validation dataset is {}. "
                    "{} "
                    "The GNN model predicts that molecule-{} {} "
                    "with a probability of {:.4f}. "
                    "In fact, molecule-{} {}.",
    "Rag-Valid-S-EP": "The SMILES string of molecule-{} in the validation dataset is {}. "
                      "The GNN model predicts that molecule-{} {} "
                      "with a probability of {:.4f}. "
                      "In fact, molecule-{} {}. "
                      "Insightful explanation: {}",
    "Rag-Valid-SD-EPD": "The SMILES string of molecule-{} in the validation dataset is {}. "
                        "{} "
                        "The GNN model predicts that molecule-{} {} "
                        "with a probability of {:.4f}. "
                        "In fact, molecule-{} {}. "
                        "Insightful explanation: {}",
    "Rag-Train-S": "The SMILES string of molecule-{} in the training dataset is {} and it {}.",
    "Rag-Train-SD": "The SMILES string of molecule-{} is {}. "
                    "{} "
                    "The molecule-{} {}.",
    "Rag-Train-S-EP": "The SMILES string of molecule-{} in the training dataset is {} and it {} {}. "
                      "Insightful explanation: {}",
    "Rag-Train-SD-EPD": "The SMILES string of molecule-{} in the training dataset is {}. "
                        "{} "
                        "The molecule-{} {}. "
                        "Insightful explanation: {}",
}
cls_rag_message_template = {
    "Rag-Valid-S": "The SMILES string of molecule-{} in the test dataset is {}. "
                   "The GNN model predicts that molecule-{} {} "
                   "with a probability of {:.4f}. "
                   "Predict whether molecule-{} {} and provide insightful explanations "
                   "by incorporating the prediction given by the GNN model and relevant knowledge. "
                   "Answer this question in the format: "
                   "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Valid-SD": "The SMILES string of molecule-{} in the test dataset is {}. "
                    "{} "
                    "The GNN model predicts that molecule-{} {} "
                    "with a probability of {:.4f}. "
                    "Predict whether molecule-{} {} and provide insightful explanations "
                    "by incorporating the prediction given by the GNN model and relevant knowledge. "
                    "Answer this question in the format: "
                    "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Valid-S-EP": "The SMILES string of molecule-{} in the test dataset is {}. "
                      "The GNN model predicts that molecule-{} {} "
                      "with a probability of {:.4f}. "
                      "Predict whether molecule-{} {} and provide insightful explanations "
                      "by incorporating the prediction given by the GNN model and relevant knowledge. "
                      "Answer this question in the format: "
                   "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Valid-SD-EPD": "The SMILES string of molecule-{} in the test dataset is {}. "
                        "{} "
                        "The GNN model predicts that molecule-{} {} "
                        "with a probability of {:.4f}. "
                        "Predict whether molecule-{} {} and provide insightful explanations "
                        "by incorporating the prediction given by the GNN model and relevant knowledge. "
                        "Answer this question in the format: "
                        "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Train-S": "The SMILES string of molecule-{} in the test dataset is {}. "
                   "The GNN model predicts that molecule-{} {} "
                   "with a probability of {:.4f}. "
                   "Predict whether molecule-{} {} and provide insightful explanations "
                   "by incorporating the prediction given by the GNN model and relevant knowledge. "
                   "Answer this question in the format: "
                   "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Train-SD": "The SMILES string of molecule-{} in the test dataset is {}. "
                    "{} "
                    "The GNN model predicts that molecule-{} {} "
                    "with a probability of {:.4f}. "
                    "Predict whether molecule-{} {} and provide insightful explanations "
                    "by incorporating the prediction given by the GNN model and relevant knowledge. "
                    "Answer this question in the format: "
                    "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Train-S-EP": "The SMILES string of molecule-{} in the test dataset is {}. "
                      "The GNN model predicts that molecule-{} {} "
                      "with a probability of {:.4f}. "
                      "Predict whether molecule-{} {} and provide insightful explanations "
                      "by incorporating the prediction given by the GNN model and relevant knowledge. "
                      "Answer this question in the format: "
                      "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Train-SD-EPD": "The SMILES string of molecule-{} in the test dataset is {}. "
                        "{} "
                        "The GNN model predicts that molecule-{} {} "
                        "with a probability of {:.4f}. "
                        "Predict whether molecule-{} {} and provide insightful explanations "
                        "by incorporating the prediction given by the GNN model and relevant knowledge. "
                        "Answer this question in the format: "
                        "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Step-S": [
        "The SMILES string of molecule-{} is {}. "
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: "
        "Prediction: <True or False>; Explanation: <text>.",
        #
        "Your prediction differs from the GNN model's prediction. "
        "The GNN model predicts that molecule-{} {} with a probability of {:.4f}, "
        "and the {} score of the GNN model on the training dataset is {:.4f}, "
        "on the validation dataset is {:.4f}. "
        "Please conduct a self-check on your thought process. "
        "Examine your reasoning carefully, and search for additional relevant information "
        "to validate or refine your current understanding. "
        "If there are uncertainties or areas where your initial thoughts "
        "might be incomplete or inaccurate, acknowledge them. "
        "Provide your refined response after considering additional information. "
        "Answer the question in the format: "
        "Prediction: <True or False>; Explanation: <text>."
    ],
    "Rag-Ranked-Train-S": "The SMILES string of molecule-{} in the test dataset is {}. "
                          "The GNN model predicts that molecule-{} {} "
                          "with a probability of {:.4f}. "
                          "Predict whether molecule-{} {} and provide insightful explanations "
                          "by incorporating the prediction given by the GNN model and relevant knowledge. "
                          "Answer this question in the format: "
                          "Prediction: <True or False>; Explanation: <text>.",
    "Rag-Ranked-Step-Train-S": [
        "The SMILES string of molecule-{} is {}. "
        "Predict whether molecule-{} {} and provide insightful explanations "
        "by incorporating the given information in the upload file and relevant knowledge. "
        "Answer this question in the format: "
        "Prediction: <True or False>; Explanation: <text>.",
        #
        "The GNN model predicts that molecule-{} {} with a probability of {:.4f}. "
        "You can find more relevant information in the uploaded file. "
        "Please conduct a self-check on your last answer and you thought process. "
        "Provide your refined response after considering additional information. "
        "Answer the question in the format: "
        "Prediction: <True or False>; Explanation: <text>."
    ],
}
cls_rag_check_message_template = {
    "double-check": "Your prediction differs from the GNN model's prediction, "
                    "and the {} score of the GNN model on the training dataset is {:.4f}, "
                    "on the validation dataset is {:.4f}. "
                    "Please conduct a self-check on your thought process. "
                    "Examine your reasoning carefully, and search for additional relevant information "
                    "to validate or refine your current understanding. "
                    "If there are uncertainties or areas where your initial thoughts "
                    "might be incomplete or inaccurate, acknowledge them. "
                    "Provide your refined response after considering additional information. "
                    "Answer the question in the format: "
                    "Prediction: <True or False>; Explanation: <text>.",
    "check-irresolute": "You altered your prediction, "
                    "please conduct a self-check on your thought process. "
                    "Examine your reasoning carefully, and search for additional relevant information "
                    "to validate or refine your current understanding. "
                    "If there are uncertainties or areas where your initial thoughts "
                    "might be incomplete or inaccurate, acknowledge them. "
                    "Provide your refined response after considering additional information. "
                    "Answer the question in the format: "
                    "Prediction: <True or False>; Explanation: <text>.",
}
cls_rag_warm_up_message_template = {
    "train": [
        "Identify patterns in the molecule structures might determine whether it {}.",
        "List all molecules in the training dataset that {}.",
        "Reason about why these molecules {}.",
        "List all molecules in the training dataset that do not {}.",
        "Reason about why these molecules do not {}.",
    ],
    "valid": [
        "List all molecules in the validation dataset that {}.",
        "Reason about why these molecules {}.",
        "List all molecules in the validation dataset that do not {}.",
        "Reason about why these molecules do not {}.",
        "List all molecules in the validation dataset that are correctly predicted by the GNN model.",
        "What patterns in the molecule structures might contribute to correct predictions?",
        "List all molecules in the validation dataset that are incorrectly predicted by the GNN model.",
        "What patterns in the molecule structures might contribute to incorrect predictions?",
    ],

    # "format": "Your answer does not follow the user's answer format requirements, "
    #           "give a refined answer after carefully considering it."
}


reg_rag_knowledge_instruction = (
    "You are an AI molecule property analysis assistant "
    "specializing in correcting predictions made by Graph Neural Networks (GNNs). "
    "The prediction is about the {} of a molecule. "
    "You can only give one reply for each conversation turn. "
    "Strictly follow the user's answer format requirements carefully and to the letter."
)
reg_rag_knowledge_template = {
    "RagSD": "The molecule-{}'s SMILES string is {}. "
             "{} "
             "This molecule's prediction given by the GNN model is {:.4f}. "
             "The real value about the {} of the molecule is {:.4f}."
}
reg_rag_message_template = {
    "RagSD": "The molecule-{}'s SMILES string is {}.\n"
             "{}\n"
             "This molecule-{}'s prediction given by the GNN model is {} "
             "with predicted probability {:.4f}.\n"
             "Provide corrected prediction. Answer this question in the format: "
             "Prediction: <number>.",
}


cls_explainer_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in explaining predictions made by Graph Neural Networks (GNNs). "
               "The prediction is about whether the molecule {}."
}
cls_explainer_template = {
    "EP": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "The molecule-{}'s prediction given by GNNs is {} "
                       "with predicted probability {:.4f}. "
                       "The true label of the molecule-{} is {}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
    "EPD": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "{} "
                       "The molecule-{}'s prediction given by GNNs is {} "
                       "with predicted probability {:.4f}. "
                       "The true label of the molecule-{} is {}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ]
}

reg_explainer_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in explaining predictions made by Graph Neural Networks (GNNs). "
               "The prediction is about the {} of a molecule."
}
reg_explainer_template = {
    "EP": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "The molecule-{}'s prediction given by GNNs is {:.4f}. "
                       "The true value of the molecule-{} is {:.4f}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
    "EPD": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "{} "
                       "The molecule-{}'s prediction given by GNNs is {:.4f}. "
                       "The true value of the molecule-{} is {:.4f}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
}


cls_corrector_instruction = {
    "role": "system",
    "content": "You are an expert in Molecules and Graph Machine Learning, "
               "specializing in correcting predictions made by Graph Neural Networks (GNNs)."
}
cls_corrector_message_template = {
    "CorrFS": [
        cls_corrector_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "Predict whether molecule-{} {} and provide insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; Explanation: <text>."
        }
    ],
    "CorrFS_knowledge": [
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "Predict whether molecule-{} {}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>."
        },
        {
            "role": "assistant",
            "content": "Prediction: {}."
        },
    ],

    "CorrFSD": [
        cls_corrector_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "{} "
                       "Predict whether molecule-{} {} and provide insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; Explanation: <text>."
        }
    ],
    "CorrFSD_knowledge": [
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "{} "
                       "Predict whether molecule-{} {}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>."
        },
        {
            "role": "assistant",
            "content": "Prediction: {}."
        },
    ],
}


reg_corrector_instruction = {
    "role": "system",
    "content": "You are an expert in Graph Machine Learning, "
               "specializing in correcting predictions made by Graph Neural Networks (GNNs). "
               "The prediction is about the {} of a molecule."
}
reg_corrector_message_template = {
    "CorrFS": [
        reg_corrector_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}, "
                       "and its prediction given by the GNN model is {:.4f}. "
                       "Provide corrected prediction and insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],
    "CorrFS_knowledge": [
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}, "
                       "and its prediction given by the GNN model is {:.4f}. "
                       "Provide corrected prediction.",
        },
        {
            "role": "assistant",
            "content": "Prediction: {:.4f}."
        },
    ],

    "CorrFSD": [
        reg_corrector_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}, "
                       "and its prediction given by the GNN model is {}. "
                       "{} "
                       "Provide corrected prediction and insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],
    "CorrFSD_knowledge": [
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}, "
                       "and its prediction given by the GNN model is {}. "
                       "{} "
                       "Provide corrected prediction.",
        },
        {
            "role": "assistant",
            "content": "Prediction: {:.4f}."
        },
    ],
}


predictor_instruction = {
    "role": "system",
    "content": "You are an expert in biomedicine and chemistry, specializing in molecules."
}
cls_predictor_message_template = {
    "IF": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Provide detailed insights on the molecule with the SMILES string {}. "
                       "I am particularly interested in understanding whether the molecule {}.",
        }
    ],

    "IFD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Provide detailed insights on the molecule with the SMILES string {}. "
                       "{} "
                       "I am particularly interested in understanding whether the molecule {}.",
        }
    ],

    "IP": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>.",
        }
    ],

    "IPD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <True or False>.",
        }
    ],

    "IE": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; "
                       "Explanation: <text>.",
        }
    ],

    "IED": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; "
                       "Explanation: <text>.",
        }
    ],

    "FS": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; "
                       "Explanation: <text>.",
        }
    ],
    "FS_knowledge_pos": [
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}.",
        },
        {
            "role": "assistant",
            "content": "Yes."
        },
    ],
    "FS_knowledge_neg": [
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}.",
        },
        {
            "role": "assistant",
            "content": "No."
        },
    ],

    "FSD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; "
                       "Explanation: <text>.",
        }
    ],
    "FSD_knowledge_pos": [
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "{}",
        },
        {
            "role": "assistant",
            "content": "Yes."
        },
    ],
    "FSD_knowledge_neg": [
        {
            "role": "user",
            "content": "Predict whether the molecule with the SMILES string {} "
                       "{}. "
                       "{}",
        },
        {
            "role": "assistant",
            "content": "No."
        },
    ],
}
reg_predictor_message_template = {
    "IF": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Provide detailed insights on the molecule with the SMILES string {}. "
                       "I am particularly interested in predicting its "
                       "{}.",
        }
    ],

    "IFD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Provide detailed insights on the molecule with the SMILES string {}. "
                       "{} "
                       "I am particularly interested in predicting its "
                       "{}.",
        }
    ],

    "IP": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "Answer this question in the format: "
                       "Prediction: <number>.",
        }
    ],

    "IPD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <number>.",
        }
    ],

    "IE": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],

    "IED": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],

    "FS": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],
    "FS_knowledge": [
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}.",
        },
        {
            "role": "assistant",
            "content": "{}."
        },
    ],

    "FSD": [
        predictor_instruction,
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "{} "
                       "Answer this question in the format: "
                       "Prediction: <number>; "
                       "Explanation: <text>.",
        }
    ],
    "FSD_knowledge": [
        {
            "role": "user",
            "content": "Predict the {} "
                       "of the molecule with the SMILES string {}. "
                       "{}",
        },
        {
            "role": "assistant",
            "content": "{}."
        },
    ],
}

