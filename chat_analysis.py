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

############################################################
#-----              CLASS DECLARATION            ----------#
############################################################

class MessageReader:
    def __init__(self, stop_words_file, chain_length=2, limit_end_message=12, absolute_end_message=20, start_string='__START__', end_string='__END__', threshold=1, names=[], skip=[], txt_names=[], json_names=[], num_files=10, build=False):
        # stop words
        self.stop_words = []
        if (os.path.isfile(stop_words_file)):
            self.read_stop_words(stop_words_file)
        
        # markov chain
        self.chain_length = chain_length
        self.start_string = start_string
        self.end_string = end_string
        self.limit_end_message = limit_end_message
        self.absolute_end_message = absolute_end_message + 2 # to account for start/end strings
        self.markov_chain = {}
        self.markov_context = {}
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
    #-----                READ FILES                 ----------#
    ############################################################

    # Retrieve the alphanumeric characters in lowercase
    def tokenise(self, word):
        return ''.join(ch for ch in word.lower() if ch.isalnum() or ch in emoji.UNICODE_EMOJI or ch=='-' or ch =='\'')

    def format_markov_chain_file(self, file_index):
        return "{}/chain_{}.json".format(self.markov_dir, file_index)

    def format_markov_context_file(self, file_index):
        return "{}/context_{}.json".format(self.markov_dir, file_index)

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

    # Given a set of filenames (txt and/or json formats) parses the messages for all
    # Outputs statistics files and  builds a markov chain
    def read_all_messages(self):
        # read all txt files
        for name in self.txt_names:
            self.read_all_messages_in_dir('./{}'.format(name), json=False)
            self.write_stat_text_files(name)
            self.reset()
        
        # read all json files
        for name in self.json_names:
            self.read_all_messages_in_dir('./{}'.format(name))
            self.write_stat_text_files(name)
            self.reset()

        # write metadata
        self.write_markov_metadata()

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
                    current_messages = previous_messages
                    previous_messages = []
                    current_sender = sender_name

                # Message counts
                self.total_no_messages += 1
                self.messages_per_person[sender_name] += 1
                self.analyse_content(split_content, sender_name)

    ############################################################
    #-----              ANALYSIS FUNCTIONS           ----------#
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
    
    # update current context to files
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

    def generate_message(self):
        print('Generating sentence using chain length of {}'.format(self.chain_length))
        # Generate first word and add to the message
        message = [self.start_string]
        self.markov_chain = self.load_json_dict(self.format_markov_chain_file('start'))
        start = self.markov_chain[self.start_string]

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

    # Analyses the frequencies of terms, bigrams and trigrams given a list of words
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

        # Bigram frequency analysis
        # Ignores frequencies of less than 4 (nonsensical phrases)
        i = 0
        while(i < len(split_content)-2):
            bigram = ' '.join([self.tokenise(w) for w in split_content[i:i+2] if self.tokenise(w) not in self.skip])
            i += 1
            if(len(bigram) > 0):
                if (bigram not in self.bigram_total_term_frequency):
                    self.bigram_total_term_frequency[bigram] = 0
                    self.bigram_term_frequency_names[bigram] = {}
                    for name in self.names:
                        self.bigram_term_frequency_names[bigram][name] = 0
                self.bigram_total_term_frequency[bigram] += 1
                if(self.bigram_total_term_frequency[bigram] >= self.threshold):
                    self.bigram_phrases += 1
                self.bigram_term_frequency_names[bigram][sender_name] += 1

        # Trigram frequency analysis
        # Ignores frequencies of less than 4 (nonsensical phrases)
        i = 0
        while(i < len(split_content)-3):
            trigram = ' '.join([self.tokenise(w) for w in split_content[i:i+3] if self.tokenise(w) not in self.skip])
            i += 1
            if(len(trigram) > 0):
                if (trigram not in self.trigram_total_term_frequency):
                    self.trigram_total_term_frequency[trigram] = 0
                    self.trigram_term_frequency_names[trigram] = {}
                    for name in self.names:
                        self.trigram_term_frequency_names[trigram][name] = 0
                self.trigram_total_term_frequency[trigram] += 1
                if(self.trigram_total_term_frequency[trigram] >= self.threshold):
                    self.trigram_phrases += 1
                self.trigram_term_frequency_names[trigram][sender_name] += 1

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
        markov_metadata = {}
        markov_metadata['chain_length'] = self.chain_length
        markov_metadata['total_message_count'] = self.markov_message_count
        markov_metadata['total_word_count'] = self.markov_word_count
        markov_metadata['start_date'] = self.markov_start_date.format('YYYY-MM-DD HH:mm:ss')
        markov_metadata['end_date'] = self.markov_end_date.format('YYYY-MM-DD HH:mm:ss')
        markov_metadata['last_updated'] = self.updated.format('YYYY-MM-DD HH:mm:ss')
        markov_metadata['files'] = self.filenames

        self.write_dict_json(markov_metadata, '{}/metadata.json'.format(self.markov_dir))

    # Writes the metadata file in a readable text format
    def write_metadata(self, friend):
        dir_name = './{}_stats'.format(friend)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # Write Metadata File
        with open('{}/{}_metadata.txt'.format(dir_name, friend), 'w') as f_write:
            f_write.write(str('Last Updated: %s' % self.updated.format('YYYY-MM-DD HH:mm:ss')))
            f_write.write('\n')

            f_write.write(str('Vocabulary Size: %d' % self.vocabulary_size))
            f_write.write('\n')
            
            f_write.write(str('Total Word Count: %d' % self.total_word_count))
            f_write.write('\n')

            for name in self.names:
                f_write.write(str('\t%s: %d' % (name, self.sum_message_length[name])))
                f_write.write('\n')

            f_write.write(str('Total Number of Messages: %d' % self.total_no_messages))
            f_write.write('\n')

            for name in self.names:
                f_write.write(str('\t%s: %d' % (name, self.messages_per_person[name])))
                f_write.write('\n')

            f_write.write('Average Message Length:\n')
            
            for name in self.names:
                f_write.write(str('\t%s: %d' % (name, self.average_message_length[name])))
                f_write.write('\n')

            f_write.write(str('Number of Bigram Phrases: %d' % self.bigram_phrases))
            f_write.write('\n')

            f_write.write(str('Number of Trigram Phrases: %d' % self.trigram_phrases))
            f_write.write('\n')

            if self.start_date is not None: 
                f_write.write(str('Start Date: %s' % self.start_date.format('YYYY-MM-DD')))
                f_write.write('\n')

            if self.end_date is not None:
                f_write.write(str('End Date: %s' % self.end_date.format('YYYY-MM-DD')))
                f_write.write('\n')

            if self.start_date and self.end_date is not None:
                diff = self.end_date - self.start_date
                f_write.write(str('Total duration: %s' % diff))
                f_write.write('\n')
                
    # Writes out the message stats in a readable text format
    def write_stat_text_files(self, friend):
        dir_name = './{}_stats'.format(friend)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        self.write_metadata(friend)

        # Write Single Term Frequency file
        with open('{}/{}_stats.txt'.format(dir_name, friend), 'w') as f_write:
            f_write.write(str('Vocabulary Size: %d' % self.vocabulary_size))
            f_write.write('\n')

            f_write.write(str('Total Number of Messages: %d' % self.total_no_messages))
            f_write.write('\n')

            for name in self.names:
                f_write.write(str('%s: %d' % (name, self.messages_per_person[name])))
                f_write.write('\n')

            f_write.write('\n')
            f_write.write('WORD FREQUENCIES')
            f_write.write('\n\n')

            sort_by_freq = sorted(self.total_term_frequency.items(),key=operator.itemgetter(1),reverse=True)
            sorted_term_frequency = collections.OrderedDict(sort_by_freq)
            for word, frequency in sorted_term_frequency.items():
                f_write.write(str('%s: %s'% (word.encode('utf8'),frequency)))
                f_write.write('\n')
                for name in self.names:
                    f_write.write('\t')
                    f_write.write(str('%s: %s'% (name,self.term_frequency_names[word][name])))
                    f_write.write('\n')

        # Write Bigram Frequency file
        with open('{}/{}_bigram_stats.txt'.format(dir_name, friend), 'w') as f_write:
            f_write.write('BIGRAM PHRASE FREQUENCIES')
            f_write.write('\n\n')
            f_write.write(str('Number of Bigram Phrases: %d' % self.bigram_phrases))
            f_write.write('\n')

            sort_by_freq = sorted(self.bigram_total_term_frequency.items(),key=operator.itemgetter(1),reverse=True)
            sorted_bigram_term_frequency = collections.OrderedDict(sort_by_freq)
            for word, frequency in sorted_bigram_term_frequency.items():
                if(frequency >= self.threshold):
                    f_write.write(str('%s: %s'% (word.encode('utf8'),frequency)))
                    f_write.write('\n')
                    for name in self.names:
                        f_write.write('\t')
                        f_write.write(str('%s: %s'% (name,self.bigram_term_frequency_names[word][name])))
                        f_write.write('\n')

        # Write Trigram Frequency file
        with open('{}/{}_trigram_stats.txt'.format(dir_name, friend), 'w') as f_write:
            f_write.write('TRIGRAM PHRASE FREQUENCIES')
            f_write.write('\n\n')
            f_write.write(str('Number of Trigram Phrases: %d' % self.trigram_phrases))
            f_write.write('\n')

            sort_by_freq = sorted(self.trigram_total_term_frequency.items(),key=operator.itemgetter(1),reverse=True)
            sorted_trigram_term_frequency = collections.OrderedDict(sort_by_freq)
            for word, frequency in sorted_trigram_term_frequency.items():
                if(frequency >= self.threshold):
                    f_write.write(str('%s: %s'% (word.encode('utf8'),frequency)))
                    f_write.write('\n')
                    for name in self.names:
                        f_write.write('\t')
                        f_write.write(str('%s: %s'% (name,self.trigram_term_frequency_names[word][name])))
                        f_write.write('\n')
# FULL BUILD
reader = MessageReader('englishST.txt', chain_length=3, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'], txt_names=['Gina'], json_names= ['Melissa', 'Malavika', 'Bogdan', 'Meredith', 'Ryan'], build=True)

# JSON ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, json_names= ['Melissa', 'Malavika', 'Bogdan', 'Meredith', 'Ryan'], build=True)

# TEXT ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'], txt_names=['Gina'], build=True)

# MESSAGE GENERATION ONLY TESTING
#reader = MessageReader('englishST.txt', chain_length=3, names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'])
print(reader.generate_message())