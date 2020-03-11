#Import Statements
import json
import os
import operator
import collections
import arrow
import emoji

############################################################
#-----              CLASS DEFINITION             ----------#
############################################################

class MessageReader:
    def __init__(self, stop_words_file, chain_length = 2, limit_end=8, absolute_end=15, start_string =  '__START__', end_string='__END__', threshold=1, names=[], skip=[]):
        # stop words
        self.stop_words = []
        if (os.path.isfile(stop_words_file)):
            self.read_stop_words(stop_words_file)
        
        # markov chain
        self.chain_length = chain_length
        self.start_string = start_string
        self.end_string = end_string
        self.markov_chain = {}
        self.markov_word_count = 0
        self.markov_message_count = 0

        # total counts
        self.total_no_messages = 0
        self.vocabulary_size = 0
        self.total_word_count = 0
        self.bigram_phrases = 0
        self.trigram_phrases = 0
        self.threshold = threshold

        # text specific
        self.names = names
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
        self.updated = arrow.utcnow()

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

    # Reads all the messages in a directory
    def read_all_messages(self, message_dir, json=True):
        if os.path.isdir(message_dir):
            message_files = os.listdir(message_dir)
        else:
            print("Invalid directory provided: {}".format(message_dir))
            return
        for message_file in message_files:
            file = '{}/{}'.format(message_dir, message_file)
            if not message_file.startswith('.') and os.path.isfile(file):
                print('Reading ',file)
                if json:
                    self.read_messages_json(file)
                else:
                    self.read_messages_txt(file)
            else:
                print("Invalid file provided: {}".format(file))
                continue

        for name in self.names:
            self.total_word_count += self.sum_message_length[name]
            self.average_message_length[name] = int(self.sum_message_length[name]/self.messages_per_person[name])

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
        for message in messages:
            
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
            if "content" in message:
                content = message["content"]

                # Message counts
                self.total_no_messages += 1
                self.messages_per_person[sender_name] += 1

                split_content = content.split()
                self.build_markov_chain(split_content)
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
                            if(sender_name[-1] == ':'):
                                sender_name = sender_name[:-1]
                        else:
                            sender_name = split_content[3]
                            split_content = split_content[4:]
                            if(sender_name[-1] == ':'):
                                sender_name = sender_name[:-1]

                # Message counts
                self.total_no_messages += 1
                self.messages_per_person[sender_name] += 1
                self.build_markov_chain(split_content)
                self.analyse_content(split_content, sender_name)

    ############################################################
    #-----              ANALYSIS FUNCTIONS           ----------#
    ############################################################

    # Build markov chain with context        
    def build_markov_chain(self, split_content):
        # tokenise words and remove links
        words = [self.tokenise(w) for w in split_content if len(self.tokenise(w)) > 0 and not self.tokenise(w).startswith('http')]
        if len(words) > self.chain_length:
            # update counts
            self.markov_message_count += 1
            self.markov_word_count += len(words)

            # add start string to markov chain and first word
            if self.start_string not in self.markov_chain:
                self.markov_chain[self.start_string] = {}
            first_word = words[0]
            if first_word not in self.markov_chain[self.start_string]:
                self.markov_chain[self.start_string][first_word] = 0
            self.markov_chain[self.start_string][first_word] += 1

            # add start and end string to the sentence
            words.insert(0, self.start_string)
            words.append(self.end_string)

            # find the full phrase being of length chain_length+1
            # where the context is the first chain_length words and the next_word is the last word in the phrase
            # update markov chain frequencies
            for i in range(len(words) - self.chain_length):
                phrase = words[i:i+self.chain_length+1]
                context = ' '.join(phrase[0:self.chain_length])
                next_word = phrase[self.chain_length]

                if context not in self.markov_chain:
                    self.markov_chain[context] = {}
                if next_word not in self.markov_chain[context]:
                    self.markov_chain[context][next_word] = 0
                self.markov_chain[context][next_word] += 1

    #TODO Generate messages using markov chain
    def generate_message(self):
        # don't forget limiting length! and possibly including a lower level threshold
        pass    

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
    def write_markov_txt(self):
        with open('markov_chain.txt', 'w') as f_write:
            f_write.write('MARKOV CHAIN')
            f_write.write('\n\n')

            f_write.write(str('Total Message Count: %d' % self.markov_message_count))
            f_write.write('\n')

            f_write.write(str('Total Word Count: %d' % self.markov_word_count))
            f_write.write('\n\n')

            for context, next_word in self.markov_chain.items():
                for word, frequency in next_word.items():
                    f_write.write(str('%s: %s, %d'% (context, word, frequency)))
                    f_write.write('\n')

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

reader = MessageReader('englishST.txt', names=['Gina', 'Sophia'], skip=['Bolognesi', 'Singh'])
reader.read_all_messages('./Gina',json=False)
reader.write_stat_text_files('Gina')
reader.reset()

names = ['Melissa', 'Malavika', 'Bogdan', 'Meredith', 'Ryan']
for name in names:
    reader.read_all_messages('./{}'.format(name))
    reader.write_stat_text_files(name)
    reader.write_stat_json_files(name)
    reader.reset()

reader.write_markov_txt()
reader.write_dict_json(reader.markov_chain, 'markov.json')

#reader = MessageReader('englishST.txt')
#sentence = 'Hello everyone, I love you all'
#reader.build_markov_chain(sentence.split())
#reader.write_markov_txt()
#reader.write_dict_json(reader.markov_chain, 'markov.json')