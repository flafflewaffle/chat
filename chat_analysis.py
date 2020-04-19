#Import Statements
import json
import os
import operator
import collections
import hashlib
import itertools
import arrow
import emoji
import random
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 

############################################################
#-----              CLASS DECLARATION            ----------#
############################################################

class MessageReader:
    def __init__(self, stop_words_file, chain_length=2, limit_end_message=12, absolute_end_message=20, threshold=1, names=[], skip=[], txt_names=[], json_names=[], num_files=10, build=False):
        # pre processing
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = []
        if (os.path.isfile(stop_words_file)):
            self.read_stop_words(stop_words_file)
        
        # markov chain
        self.chain_length = chain_length
        self.start_string = '__START__'
        self.end_string = '__END__'
        self.limit_end_message = limit_end_message
        self.absolute_end_message = absolute_end_message + 2 # to account for start/end strings
        self.markov_chain = {}
        self.markov_context = {}
        self.topics = {}
        self.reactions = {}
        self.markov_word_count = 0
        self.markov_message_count = 0
        self.markov_start_date = None
        self.markov_end_date = None
        self.build = build
        self.filenames = []

        # total counts
        self.total_no_messages = 0
        self.vocabulary_size = 0
        self.total_word_count = 0
        self.bigram_phrases = 0
        self.trigram_phrases = 0
        self.threshold = threshold

        # filenames specified by txt (whatsapp) or json (facebook)
        self.names = names
        self.txt_names = txt_names
        self.json_names = json_names
        self.num_files = num_files
        self.skip = [self.tokenise(w) for w in skip]

        # date time 
        self.start_date = None
        self.end_date = None
        self.updated = arrow.utcnow()

        # frequency counts
        self.messages_per_person = {}
        self.sum_message_length = {}
        self.average_message_length = {}
        self.total_term_frequency = {}
        self.term_frequency_names = {}
        self.bigram_total_term_frequency = {}
        self.bigram_term_frequency_names = {}
        self.trigram_total_term_frequency = {}
        self.trigram_term_frequency_names = {}
        
        # create a markov chain if it doesn't already exist
        self.markov_dir = './markov_chain_{}'.format(self.chain_length)
        if not os.path.isdir(self.markov_dir):
            os.mkdir(self.markov_dir)

        # load markov metadata from existing json and metadata information if not building
        if not self.build:    
            markov_metadata_file = '{}/metadata.json'.format(self.markov_dir)
            if os.path.isfile(markov_metadata_file):
                markov_metadata = self.load_json_dict(markov_metadata_file)
                self.markov_message_count = markov_metadata['total_message_count']
                self.markov_word_count = markov_metadata['total_word_count']
                self.markov_start_date = arrow.get(markov_metadata['start_date'])
                self.markov_end_date = arrow.get(markov_metadata['end_date'])
                self.filenames = markov_metadata['files']
            if os.path.isfile('reactions.json'):
                self.reactions = self.load_json_dict('reactions.json')
            if os.path.isfile('topics.json'):
                self.topics = self.load_json_dict('topics.json')
        else:
            print('Rebuilding Markov Chain with Chain Length {}'.format(self.chain_length))
        
        # automatically call read all messages if txt_names and json_names are populated
        if (txt_names or json_names) and self.build:
            # delete all files in markov model directory if rebuilding
            for filename in os.listdir(self.markov_dir):
                file_path = os.path.join(self.markov_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            self.read_all_messages()
            
    # Read Stop Words
    def read_stop_words(self, stop_words_file):
        with open(stop_words_file, 'r') as f:
            print('Reading Stop Words')
            for line in f:
                for word in line.split():
                    self.stop_words.append(word.lower())

    # Reset all variables
    def reset(self):
        self.total_no_messages = 0
        self.vocabulary_size = 0
        self.total_word_count = 0
        self.bigram_phrases = 0
        self.trigram_phrases = 0

        self.start_date = None
        self.end_date = None

        self.messages_per_person = {}
        self.sum_message_length = {}
        self.average_message_length = {}
        self.total_term_frequency = {}
        self.term_frequency_names = {}
        self.bigram_total_term_frequency = {}
        self.bigram_term_frequency_names = {}
        self.trigram_total_term_frequency = {}
        self.trigram_term_frequency_names = {}

    ############################################################
    #-----              HELPER FUNCTIONS             ----------#
    ############################################################

    # Retrieve the alphanumeric characters in lowercase
    def tokenise(self, word):
        return ''.join(ch for ch in word.lower() if ch.isalnum() or ch in emoji.UNICODE_EMOJI or ch=='-' or ch =='\'')

    def preprocess(self, terms):
        tokens = {}
        for token in terms.split():
            token = self.tokenise(token)
            if len(token) > 3 and token not in self.stop_words and token not in self.skip:
                stem = self.stemmer.stem(self.lemmatizer.lemmatize(token))
                tokens[stem] = token
        return tokens

    def format_markov_chain_file(self, file_index):
        return "{}/chain_{}.json".format(self.markov_dir, file_index)

    def format_markov_context_file(self, file_index):
        return "{}/context_start.json".format(self.markov_dir)

    # Returns the file index for a term
    def hash_term(self, term):
        hash = hashlib.md5(term.encode())
        return int(hash.hexdigest(), 16) % self.num_files
    
    # Returns dict of relevant files->terms for a list of terms. Used to minimise the number of calls to open/close a file.
    def relevant_files(self, terms, formatter):
        files = {}
        for term in terms:
            # Create a special start file for the start_string otherwise use the hash function
            if term == self.start_string:
                file_no = 'start'
            else:
                file_no = self.hash_term(term)
            file_name = formatter(file_no)
            if file_name not in files:
                files[file_name] = []
            files[file_name].append(term)
        return files

    # updates files with current dictionary input 
    def update_files(self, input_dict, formatter):
        # group files to list of chain lengths
        relevant_files = self.relevant_files(input_dict.keys(), formatter)

        for context_file, chains in relevant_files.items():
            try:
                # read existing relevant file and add chain lengths to it
                relevant_dict = self.load_json_dict(context_file)

                for chain in chains:
                    if chain in relevant_dict:
                        relevant_dict[chain].update(input_dict[chain])
                    else:
                        relevant_dict[chain] = input_dict[chain]
            except (FileNotFoundError, EOFError):
                #file not found, so create a new file alltogether
                relevant_dict = {k: input_dict[k] for k in chains}

            # write out to context file
            self.write_dict_json(relevant_dict, context_file)

    ############################################################
    #-----                READ FILES                 ----------#
    ############################################################

    # Given a set of filenames (txt and/or json formats) parses the messages for all
    # Outputs statistics files and  builds a markov chain
    def read_all_messages(self):
        # read all txt files
        for name in self.txt_names:
            self.read_all_messages_in_dir('./{}'.format(name), json=False)
            self.write_friend_metadata(name)
            self.reset()
        
        # read all json files
        for name in self.json_names:
            self.read_all_messages_in_dir('./{}'.format(name))
            self.write_friend_metadata(name)
            self.reset()

        # write metadata
        self.write_markov_metadata()
        self.write_topics()

    # Reads all the messages in a directory
    def read_all_messages_in_dir(self, message_dir, json=True):
        if os.path.isdir(message_dir):
            message_files = os.listdir(message_dir)
        else:
            print("Invalid directory provided: {}".format(message_dir))
            return
        for message_file in message_files:
            file = '{}/{}'.format(message_dir, message_file)
            if not message_file.startswith('.') and os.path.isfile(file):
                print('Reading ',file)
                self.filenames.append(file)
                if json:
                    self.read_messages_json(file)
                else:
                    self.read_messages_txt(file)
            else:
                print("Invalid file provided: {}".format(file))
                continue

            # update context files and reset the local variable
            if self.markov_chain:
                self.update_files(self.markov_chain, self.format_markov_chain_file)
                self.markov_chain = {}
            
            # update context files and reset the local variable
            if self.markov_context:
                self.update_files(self.markov_context, self.format_markov_context_file)
                self.markov_context = {}
        
        # Calculate total word and average message lengths
        for name in self.names:
            self.total_word_count += self.sum_message_length[name]
            self.average_message_length[name] = int(self.sum_message_length[name]/self.messages_per_person[name])
        
        # Update start and end date for markov data
        if self.markov_start_date is not None and self.start_date is not None: 
            if self.start_date < self.markov_start_date:
                self.markov_start_date = self.start_date
        else:
            self.markov_start_date = self.start_date

        if self.markov_end_date is not None and self.end_date is not None: 
            if self.end_date > self.markov_end_date:
                self.markov_end_date = self.end_date
        else:
            self.markov_end_date = self.end_date

    # Reads a json file and saves the messages in a dictionary
    def read_json(self, filename):
        with open(filename) as json_file:
            messages = json.load(json_file)
        return messages

    # Reads and analyses the frequencies of messages (provided a json file)
    def read_messages_json(self, message_file):
        messages_json = self.read_json(message_file)
        print('Retrieving Names')
        participants = messages_json["participants"]
        self.names = [name for participant in participants for (key,name) in participant.items()]
        print('Reading Messages For, ', self.names)

        for name in self.names:
            if(name not in self.messages_per_person):
                self.messages_per_person[name] = 0
            if(name not in self.sum_message_length):
                self.sum_message_length[name] = 0
            if(name not in self.average_message_length):
                self.average_message_length[name] = 0

        messages = messages_json["messages"]
        current_sender = ''
        previous_sender = ''
        current_messages = []
        previous_messages = []

        for message in messages:
            
            reaction = ''
            if "reactions" in message:
                reaction = message["reactions"][0]["reaction"]

            # Update start and end date
            date = arrow.get(message["timestamp_ms"]/1000)
            if self.start_date is not None: 
                if date < self.start_date:
                    self.start_date = date
            else:
                self.start_date = date
            if self.end_date is not None: 
                if date > self.end_date:
                    self.end_date = date
            else:
                self.end_date = date

            # Update counts for message and calls analyse content for stats
            sender_name = message["sender_name"]

            if current_sender != sender_name:
                previous_sender = current_sender
                self.build_markov_chain(previous_messages, reaction)
                self.build_context(previous_messages, current_messages)
                self.analyse_topics(previous_messages)
                current_messages = previous_messages
                previous_messages = []
                current_sender = sender_name

            if "content" in message:
                current_message = message["content"]
                previous_messages.append(current_message)

                # Message counts
                self.total_no_messages += 1
                self.messages_per_person[sender_name] += 1

                split_content = current_message.split()
                self.analyse_content(split_content, sender_name)
    
    # Reads and analyses the frequencies of messages (provided a txt file)
    def read_messages_txt(self, message_file):
        print('Reading Messages For, ', self.names)

        for name in self.names:
            if(name not in self.messages_per_person):
                self.messages_per_person[name] = 0
            if(name not in self.sum_message_length):
                self.sum_message_length[name] = 0
            if(name not in self.average_message_length):
                self.average_message_length[name] = 0
        current_sender = ''
        previous_sender = ''
        current_messages = []
        previous_messages = []

        with open(message_file, 'r') as f_read:
            for line in f_read:
                split_content = line.split()
                if(len(split_content)==0):
                    continue
                if('Messages to this chat and calls are now secured with end-to-end encryption. Tap for more info.' in line):
                    continue
                # Date and name
                date = split_content[0]
                if('/' in date and (date[0:2].isnumeric() or date[1:3].isnumeric())):
                    if(':' in line):
                        if(split_content[2] != '-'):
                            sender_name = split_content[2]
                            split_content = split_content[3:]
                            current_message = ' '.join(split_content)
                            if(sender_name[-1] == ':'):
                                sender_name = sender_name[:-1]
                        else:
                            sender_name = split_content[3]
                            split_content = split_content[4:]
                            current_message = ' '.join(split_content)
                            if(sender_name[-1] == ':'):
                                sender_name = sender_name[:-1]
                    else:
                        current_message = ' '.join(split_content)

                previous_messages.append(current_message)

                if current_sender != sender_name:
                    previous_sender = current_sender
                    self.build_markov_chain(previous_messages, '')
                    self.build_context(previous_messages, current_messages)
                    self.analyse_topics(previous_messages)
                    current_messages = previous_messages
                    previous_messages = []
                    current_sender = sender_name

                # Message counts
                self.total_no_messages += 1
                self.messages_per_person[sender_name] += 1
                self.analyse_content(split_content, sender_name)

    ############################################################
    #-----           MARKOV CHAIN FUNCTIONS          ----------#
    ############################################################

    # Build markov chain
    def build_markov_chain(self, messages, reaction):
        for message in messages:
            # Split the message into chain lengths
            chain = self.chain_message(message)
            
            if chain:
                # update counts
                self.markov_message_count += 1
                self.markov_word_count += self.chain_length + len(chain)-2

                # add start string to markov chain and first word
                if self.start_string not in self.markov_chain:
                    self.markov_chain[self.start_string] = {}
                # first word(s) should be of length chain_length-1
                first_word = ' '.join(chain[0][1:self.chain_length])
                if first_word not in self.markov_chain[self.start_string]:
                    self.markov_chain[self.start_string][first_word] = 0
                self.markov_chain[self.start_string][first_word] += 1

                # iterate through all chain links
                for link in chain:
                    context = ' '.join(link[0:self.chain_length])
                    next_word = link[self.chain_length]

                    # update markov chain
                    if context not in self.markov_chain:
                        self.markov_chain[context] = {}
                    if next_word not in self.markov_chain[context]:
                        self.markov_chain[context][next_word] = 0
                    self.markov_chain[context][next_word] += 1
                    if reaction:
                        if reaction not in self.markov_chain[context]:
                            self.markov_chain[context][reaction] = 0
                        self.markov_chain[context][reaction] += 1

    # splits a message and returns a list of chain lengths adding the start and end strings
    def chain_message(self, message):
        chain = []
        split_content = message.split()
        # tokenise and remove links
        words = [self.tokenise(w) for w in split_content if len(self.tokenise(w)) > 0 and not self.tokenise(w).startswith('http') and self.tokenise(w) not in self.skip]
        if len(words) > self.chain_length-2:            
            # add start and end string to the sentence
            words.insert(0, self.start_string)
            words.append(self.end_string)

            # find the full phrase being of length chain_length+1
            # where the context is the first chain_length words and the next_word is the last word in the phrase
            # update markov chain frequencies
            for i in range(len(words) - self.chain_length):
                phrase = words[i:i+self.chain_length+1]
                chain.append(phrase)
        return chain

    # build context model
    def build_context(self, previous_messages, current_messages):
        current_chains = []
        previous_chains = []

        # if neither current or previous message blocks are empty split into chain lengths
        if current_messages and previous_messages:
            for message in current_messages:
                current_chains.append(self.chain_message(message))
            current_chains = list(itertools.chain.from_iterable(current_chains))

            for message in previous_messages:
                previous_chains.append(self.chain_message(message))
            previous_chains = list(itertools.chain.from_iterable(previous_chains))

        current_chains = [chain for chain in current_chains if self.start_string in chain]
        previous_chains = [chain for chain in previous_chains if self.start_string in chain]

        # Map the starts of previous messages to the start of current messages
        if current_chains and previous_chains:
            for previous_chain in previous_chains:
                previous = ' '.join(previous_chain[0:self.chain_length])
                if previous not in self.markov_context:
                    self.markov_context[previous] = {}
                
                for current_chain in current_chains:
                    current = ' '.join(current_chain[1:self.chain_length])
                    if current not in self.markov_context[previous]:
                        self.markov_context[previous][current] = 0
                    self.markov_context[previous][current] += 1
        
    def generate_message(self, input_message):
        print('Generating sentence using chain length of {}'.format(self.chain_length))
        message = [self.start_string]
        start_input = ''
        input_chain = self.chain_message(input_message)
        if input_chain:
            start_input = ' '.join(input_chain[0][0:self.chain_length])
            self.markov_context = self.load_json_dict(self.format_markov_context_file('start'))

        # if input message is in context, use that for the start
        # else use the generic markov chain start
        if start_input in self.markov_context:
            start = self.markov_context[start_input]
        else:   
            self.markov_chain = self.load_json_dict(self.format_markov_chain_file('start'))
            start = self.markov_chain[self.start_string]

        # Generate first word and add to the message
        start_words = list(start.keys())
        start_weights = list(start.values())
        # randomly select start word using the frequency of the word as weight
        first_word = random.choices(start_words, weights=start_weights)
        start_key = first_word[0].split()
        message.extend(start_key)

        i = 0
        # build while haven't reached the end_string and length of message hasn't reached absolute limit
        while((not message[-1] == self.end_string) and len(message) < self.absolute_end_message):
            # current context is the chain_length
            context = ' '.join(message[i:i+self.chain_length])
            context_index = self.hash_term(context)
            self.markov_chain = self.load_json_dict(self.format_markov_chain_file(context_index))
            if(context not in self.markov_chain):
                break
            next = self.markov_chain[context]

            # If the first length limit of the message is reached, start to favour the end_string
            if len(message) >= self.limit_end_message:
                if self.end_string in next:
                    next[self.end_string] += 10
            
            # randomly select next word using the frequency of the word as weight
            next_words = list(next.keys())
            next_weights = list(next.values())
            next_word = random.choices(next_words, weights=next_weights)
            message.extend(next_word)

            i+=1

        # Delete start and end strings and return a string
        message.pop(0)
        if message[-1] == self.end_string:
            message.pop(-1)

        return ' '.join(message).capitalize()

    # Preprocesses and analyses messages and records frequencies to model topics
    def analyse_topics(self, messages):
        for message in messages:
            preprocess = self.preprocess(message)
            tagged_tokens = nltk.pos_tag(list(preprocess.keys()))
            for (token, tag) in tagged_tokens:
                if tag == 'NN' or tag == 'VBP':
                    word = self.tokenise(preprocess[token])
                    if word not in self.topics:
                        self.topics[word] = 0
                    self.topics[word] += 1

    # Analyses the frequencies of term per person and overall vocabulary size
    def analyse_content(self, split_content, sender_name):
        self.sum_message_length[sender_name] += len(split_content)
        # Single word frequency analysis
        for word in split_content:
            term = self.tokenise(word)
            if(len(term) > 0 and term not in self.stop_words and term not in self.skip):
                if (term not in self.total_term_frequency):
                    self.total_term_frequency[term] = 0
                    self.vocabulary_size += 1
                    self.term_frequency_names[term] = {}
                    for name in self.names:
                        self.term_frequency_names[term][name] = 0
                self.total_term_frequency[term] += 1
                self.term_frequency_names[term][sender_name] += 1

    ############################################################
    #-----              WRITE OUTPUT FILES           ----------#
    ############################################################

    # Write any dictionary to json file
    def write_dict_json(self, input_dict, output_file):
        with open(output_file, 'w') as json_file:
            json.dump(input_dict, json_file)

    # loads dict from json and returns the dict to be used
    def load_json_dict(self, json_file):
        with open(json_file, 'r') as json_file:
            return json.load(json_file)

    # Writes out the message stats in json files
    def write_stat_json_files(self, friend):
        dir_name = './{}_stats'.format(friend)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        self.write_dict_json(self.total_term_frequency, '{}/{}_stats.json'.format(dir_name,friend))
        self.write_dict_json(self.bigram_total_term_frequency, '{}/{}_bigram_stats.json'.format(dir_name,friend))
        self.write_dict_json(self.trigram_total_term_frequency, '{}/{}_trigram_stats.json'.format(dir_name,friend))

    # Writes Markov chain to text file for readability
    def write_markov_metadata(self):
        metadata = {}
        metadata['chain_length'] = self.chain_length
        metadata['total_message_count'] = self.markov_message_count
        metadata['total_word_count'] = self.markov_word_count
        metadata['start_date'] = self.markov_start_date.format('YYYY-MM-DD HH:mm:ss')
        metadata['end_date'] = self.markov_end_date.format('YYYY-MM-DD HH:mm:ss')
        metadata['last_updated'] = self.updated.format('YYYY-MM-DD HH:mm:ss')
        metadata['files'] = self.filenames

        self.write_dict_json(metadata, '{}/metadata.json'.format(self.markov_dir))

    # Writes the metadata file for each message set 
    def write_friend_metadata(self, friend):
        metadata = {}
        metadata['last_updated'] = self.updated.format('YYYY-MM-DD HH:mm:ss')
        metadata['vocabulary_size'] = self.vocabulary_size
        metadata['total_word_count'] = self.total_word_count
        metadata['word_count_per_person'] = {}
        for name in self.names:
            metadata['word_count_per_person'][name] = self.sum_message_length[name]
        metadata['total_number_of_messages'] = self.total_no_messages
        metadata['messages_per_person'] = {}
        for name in self.names:
            metadata['messages_per_person'][name] = self.messages_per_person[name]
        metadata['average_message_length'] = {}
        for name in self.names:
            metadata['average_message_length'][name] = self.average_message_length[name]
        if self.start_date is not None:
            metadata['start_date'] = self.start_date.format('YYYY-MM-DD')
        if self.end_date is not None:
            metadata['end_date'] = self.end_date.format('YYYY-MM-DD')
        if self.start_date and self.end_date is not None:
            diff = self.end_date - self.start_date
            metadata['total_duration'] = str(diff)
        self.write_dict_json(metadata, '{}_metadata.json'.format(friend))

    # Write out most common topics (words) and filters past the defined threshold
    def write_topics(self):
        self.topics = dict(filter(lambda elem: elem[1] >= self.threshold, self.topics.items()))
        self.write_dict_json(self.topics, 'topics.json')
        with open('topics.txt', 'w') as f:
            sort_by_freq = sorted(self.topics.items(),key=operator.itemgetter(1),reverse=True)
            sorted_term_frequency = collections.OrderedDict(sort_by_freq)
            for word, frequency in sorted_term_frequency.items():
                if frequency >= self.threshold:
                    f.write(str('%s: %s'% (word.encode('utf8'),frequency)))
                    f.write('\n')

# FULL BUILD
reader = MessageReader('englishST.txt', chain_length=3, threshold=30, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'], txt_names=['Gina'], json_names= ['Melissa', 'Malavika', 'Bogdan', 'Meredith', 'Ryan'], build=True)

# JSON ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, json_names= ['Melissa', 'Malavika', 'Bogdan', 'Meredith', 'Ryan'], build=True)

# TEXT ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'], txt_names=['Gina'], build=True)

# MESSAGE GENERATION ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'])
print(reader.generate_message("Hope you have a wonderful day"))