from googletestclass import our_LLM_class
import pathlib
import time
start_time = time.time()

path_to_dir = pathlib.Path(__file__).parent.resolve()


LLMClass = our_LLM_class(str(path_to_dir))
LLMClass.aux_tasks = 35
LLMClass.set_message("form_4")
LLMClass.all_desc_inc['Mordred 2D descriptors'] = True
LLMClass.all_desc_inc['RDKit 2D descriptors'] = True
LLMClass.all_desc_inc['QM properties'] = True
LLMClass.NO_tournaments = 10
LLMClass.save_conversation = True
LLMClass.props_per_query = 500

LLMClass.run_properties_tournament()


# f = open("googleconversation.txt", "r")
# LLMClass.queryOpenAI(f.read())
# f.close()

print("My program took", time.time() - start_time, "to run")

