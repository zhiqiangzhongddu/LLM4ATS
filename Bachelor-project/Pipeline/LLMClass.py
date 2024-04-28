from openai import OpenAI
from collections import Counter
import random

class our_LLM_class:
    # The target task
    target_task = "level of toxicity"
    
    # The wanted format of the anwser
    anwser_format = "And can you provide the anwser in the format like like shown below?\n- anwser\n- anwser\n..."
    
    # Max number of properties to get from the LLM
    aux_tasks = 20 
    
    # The default prefix string
    prefix_str = f"Can you provide a subset of maximum size of {aux_tasks} out of the given list of properties that have the highest correlation with the {target_task} in a given molecule? Please select the properties that are the most important. {anwser_format}\n"
    
    # 45 properties per query
    props_per_query = 45
    
    # Number of tournaments
    NO_tournaments = 10
    
    # All descriptors segregated by source
    all_descriptors = {}
    
    # Whether the descriptors from a given source should be included in the tournament
    all_desc_inc = {}
    
    # Whether to print the queries and responses to a file
    save_conversation = False

    # Name of the file to store the conversation with chat-GPT
    conversation_file = "conversation.txt"
    
    # Initialize class and read all the possible properties/descriptors
    def __init__(self):
        f = open("descriptors.txt", "r", encoding="utf8")
        total = f.read().split("\n\n")
        for i in range(0,len(total),2):
            self.all_descriptors[total[i][:-1]] = total[i+1].split("\n")
            self.all_desc_inc[total[i][:-1]] = False
        f.close()
    
    # Calls chat-GPT and saves the conversation if arg is set to True
    def queryOpenAI(self, question):
        apikey = open("../apikey.txt", "r")
        client = OpenAI(
            # This is the default way to read the apikey. It can be changed depending on how the key is stored.
            api_key=apikey.read(),
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="gpt-3.5-turbo",
        )
        if self.save_conversation:
            textChata = open(self.conversation_file, "a")
            textChata.write("\nQuery\n\n"+self.__insertNewlines(question, 200)+ "\n\n" + "-"*200)
            output = chat_completion.choices[0].message.content
            textChata.write("\nAnwser\n\n"+self.__insertNewlines(output, 200)+ "\n\n" + "-"*200)
            textChata.close()
            apikey.close()
        return output
    

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
        queries = self.__get_queries("descriptors.txt", True)
        lst = ""
        for i in range(self.NO_tournaments):
            print(f'Round {i}')
            lst = lst + self.__get_final_result(queries)
        lst = lst.split("\n")[0:-1]
        print(Counter(lst))
        return Counter(lst)

    # Splits the list of properties into several quiries of size 'props_per_query'
    def __get_queries(self, descriptors : str, is_file : bool) -> list: 
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
                queries.append(self._s + query)
                query = s + ".\n"
            else:
                query = query + s + ".\n"
            counter += 1
        # append the rest of the properties
        if (counter % self.props_per_query)!=0:
            queries.append(self._s + query)
        return queries

    # Recursively calls chat-GPT until the number of suggested properties is 'aux_tasks'
    def __get_final_result(self, queries):
        extracted_responses = ""
        for i, q in enumerate(queries):    
            response = self.queryOpenAI(q)
            print(f"Query {i}")
            extracted_responses = extracted_responses + self.__extract_response(response)
        tmp = len(extracted_responses.split("\n"))
        print(f"Number of extracted responses {tmp}")
        if tmp <= self.aux_tasks + 1:
            return extracted_responses
        return self.get_final_result(extracted_responses, False)
    
    # Extracts the list of properties from the response given by chat-GPT
    def __extract_response(response : str):
        resp = response.split("\n")
        result = ""
        for s in resp:
            result = result + s[2:] + "\n"
        return result

    # Selects random properties. Does not call chat-GPT.
    def get_random_properties(self):
        props = []
        for key in self.all_descriptors:
            if self.all_desc_inc[key]:
                props += self.all_descriptors[key]
        return random.sample(props, self.aux_tasks)
    

        
        
        
        
        
        