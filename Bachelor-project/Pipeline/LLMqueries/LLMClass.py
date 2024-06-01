from openai import OpenAI
from collections import Counter
import random
from LLMqueries.query_attempts import query_attempts
import copy
import tiktoken

# List of properties that, for some reason, results in errors for all molecules
fyfy = ["total hydrophobic surface area", "total polar surface area"]

class our_LLM_class:
    # Filepath
    path = ""

    # The target task
    target_task = "level of toxicity"
    
    # The wanted format of the anwser
    anwser_format = "And can you provide the anwser in the format like like shown below? do NOT deviate from this format.\n\n- <anwser>\n- <anwser>\n- <anwser>..."
    
    # Number of auxiliary tasks to return
    aux_tasks = 5 

    # Max number of properties to get from the LLM
    max_props = 10
    
    template_set = query_attempts
    
    message_name = 'default'
    
    message = copy.deepcopy(template_set['default'])
    
    prefix_str = message[1]['content'].format(max_props, target_task, anwser_format)
    
    # token_encoding = tiktoken.get_encoding("cl100k_base")

    
    # 45 properties per query
    props_per_query = 45
    
    # Number of tournaments
    NO_tournaments = 10
    
    # All descriptors segregated by source
    all_descriptors = {}
    
    # Whether the descriptors from a given source should be included in the tournament
    all_desc_inc = {}
    
    all_desc_wo_des = {}
    
    
    
    # Whether to print the queries and responses to a file
    save_conversation = False

    # Name of the file to store the conversation with chat-GPT
    conversation_file = path+"/LLMqueries/txt_files/conversation.txt"
    
    # Whether to use properties from previous LLM runs
    prev_prop_flag = False

    # Seed to be used when selecting random properties
    random_props_seed = 42 # placeholder

    
    
    # Initialize class and read all the possible properties/descriptors
    def __init__(self, path):
        self.path = str(path)
        self.client = OpenAI(api_key=open(self.path + "/apikey.txt", "r").read())
        f = open(self.path+"/LLMqueries/txt_files/descriptors.txt", "r", encoding="utf8")
        total = f.read().split("\n\n")
        for i in range(0,len(total),2):
            self.all_descriptors[total[i][:-1]] = total[i+1].split("\n")
            self.all_desc_wo_des[total[i][:-1]] = [(e.split(":")[0]).lower() for e in total[i+1].split("\n")]
            self.all_desc_inc[total[i][:-1]] = False
        f.close()
    
    # Calls chat-GPT and saves the conversation if arg is set to True
    def queryOpenAI(self):
        # apikey = open(self.path+"apikey.txt", "r")
        # client = OpenAI(
        #     # This is the default way to read the apikey. It can be changed depending on how the key is stored.
        #     api_key=apikey.read(),
        # )
        chat_completion = self.client.chat.completions.create(
            messages=self.message,
            model="gpt-4o" #"gpt-3.5-turbo",
        )
        
        output = chat_completion.choices[0].message.content
        if self.save_conversation:
            textChata = open(self.conversation_file, "a")
            textChata.write("\nQuery\n\n"+self.__insertNewlines(self.message[1]['content'], 200)+ "\n\n" + "-"*200)
            textChata.write("\nAnwser\n\n"+self.__insertNewlines(output, 200)+ "\n\n" + "-"*200)
            textChata.close()
            
        return output
    
    def set_message(self, prefix_name):
        if prefix_name in self.template_set:
            self.message_name = prefix_name
            self.message = copy.deepcopy(self.template_set[prefix_name])
            if prefix_name not in  ['form_4', 'g_form_4', 'form_5']:
                self.prefix_str = self.message[1]['content'].format(self.max_props, self.target_task, self.anwser_format)
        else:
            print("Invalid prefix name")

    # Helper function to help format the conversation file 
    def __insertNewlines(self, text, lineLength):
        if len(text) <= lineLength:
            return text
        elif text[lineLength] != ' ':
            return self.__insertNewlines(text[:], lineLength + 1)
        else:
            return text[:lineLength] + '\n' + self.__insertNewlines(text[lineLength + 1:], lineLength)

    # Runs a tournament between several responses from chat-GPT.
    # The list of properties gets fed to chat-GPT in several round to circumvent token limit.
    def run_properties_tournament(self):
        if self.prev_prop_flag:
            return self.use_prev_prop()
        queries = self.__get_queries()
        lst = ""
        for i in range(self.NO_tournaments):
            print(f'Round {i+1}')
            lst = lst + self.__get_final_result(queries)
        lst = lst.split("\n")[0:-1]
        for i,e in enumerate(lst):
            if e.split(":")[0][-4] == "type":
                continue
            lst[i] = e.split(":")[0]
        cnt = Counter(lst)
        cnt = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))

        f = open(self.path+"/LLMqueries/txt_files/counters.txt", "a")
        f.write(f"Message template: {self.message_name}, Number of tournaments: {self.NO_tournaments}, Target task: {self.target_task}, Number of features: {self.max_props}\n\n")
        for key in cnt:
            f.write(f"{key}: {cnt[key]}\n")
        f.write("\n\n"+"-"*50+"\n\n")
        f.close()
        # Return the desired number of aux tasks
        res = []
        for key in cnt:
            res.append(key)
            if len(res) == self.aux_tasks:
                break
        return res

    # Splits the list of properties into several quiries of size 'props_per_query'
    def __get_queries(self, descriptors : str = '', is_file : bool = True) -> list: 
        props = []
        if is_file:
            for key in self.all_descriptors:
                if self.all_desc_inc[key]:
                    props += self.all_descriptors[key]
        else:
            props = descriptors.split("\n")
        queries = []
        query = ""
        counter = 0
        for s in props:
            if (counter % self.props_per_query)==0 and counter != 0:
                if self.message_name in ['form_4', 'g_form_4', 'form_5']:
                    queries.append(query_attempts[self.message_name][1]['content'].format(query,self.max_props, self.target_task,self.target_task,self.max_props))
                else: queries.append(self.prefix_str + query)
                query = s + "\n"
            else:
                query = query + s + "\n"
            counter += 1
        # append the rest of the properties
        if (counter % self.props_per_query)!=0:
            if self.message_name in ['form_4', 'g_form_4', 'form_5']:
                queries.append(query_attempts[self.message_name][1]['content'].format(query,self.max_props, self.target_task,self.target_task, self.max_props))
            else: queries.append(self.prefix_str + query)
        print(f"Number of queries: {len(queries)}")
        return queries
        
    # Recursively calls chat-GPT until the number of suggested properties is 'max_props'
    def __get_final_result(self, queries):
        extracted_responses = ""
        for i, q in enumerate(queries):    
            print(f"Query {i+1}") # , Token length: {len(self.token_encoding.encode(q))}
            self.message[1]['content'] = q
            response = self.queryOpenAI()
            extracted_responses = extracted_responses + self.__extract_response(response)[:-1]
        tmp = len(extracted_responses.split("\n"))
        print(f"Number of extracted responses {tmp}")
        if tmp <= self.max_props + 1:
            return extracted_responses
        return self.__get_final_result(self.__get_queries(extracted_responses,False))
    
    # Extracts the list of properties from the response given by chat-GPT
    def __extract_response(self,response : str):
        resp = response.split("\n")
        result = ""
        for s in resp:
            if s == '' or s[0] != "-":
                continue
            result = result + s[2:] + "\n"
        return result

    # Selects random properties. Does not call chat-GPT.
    def get_random_properties(self):
        print(self.random_props_seed)
        props = []
        for key in self.all_descriptors:
            if self.all_desc_inc[key]:
                props += self.all_descriptors[key]
        random.seed(self.random_props_seed)
        res = random.sample(props, self.aux_tasks)
        for i,e in enumerate(res):
            if e.split(":")[0][-4:] == "type":
                continue
            else:
                res[i] = e.split(":")[0]
        return res
    
    # Function that reuses previous LLM filtered properties
    def use_prev_prop(self):
        print("Using previous properties")
        file = open(self.path+"/LLMqueries/txt_files/counters.txt", "r", encoding="utf8")
        file_content = file.read()
        delimiter = ("Message template: " + self.message_name
                    + ", Number of tournaments: " + str(self.NO_tournaments)
                    + ", Target task: " + self.target_task
                    + ", Number of features: " + str(self.max_props))
        ct_split = file_content.split(delimiter)
        print("--------------------------------")
        if ct_split[0] != file_content:
            # Choose the last attempt. If there are multiple attempts using the exact same settings, the newest attempt will be selected.
            llm_query = ct_split[-1]
            # Split on double newlines to isolate the list of props from the newlines.
            llm_prop_string = llm_query.split("\n\n")
            # Turn the remaining string into a list.
            llm_prop_list = llm_prop_string[1].split("\n")
            # Loop for creating list of exact property names.
            out = []
            # Select only the first aux_tasks of properties.
            i =-1
            while len(out) < self.aux_tasks:
                i+=1
                prop = llm_prop_list[i].split(":")[0]
                if prop in fyfy: continue # Ignore uncalulatable properties
                out.append(prop)
            file.close()
            return out
        else:
            return []