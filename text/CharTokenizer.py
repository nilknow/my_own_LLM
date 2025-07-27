class CharTokenizer:
    def __init__(self, text_filepath, special_tokens=None):
        self.text_filepath = text_filepath
        # Pad: Padding Token; UNK: Unknown Token; BOS/EOS: Beginning/End Of Sentence
        self.special_tokens = special_tokens if special_tokens is not None else ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self._build_vocabulary()

    def _build_vocabulary(self):
        all_chars = set()
        try:
            with open(self.text_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    for char in line:
                        all_chars.add(char)
        except FileNotFoundError:
            print(f"Error: Text file not found at {self.text_filepath}")
            return
        except Exception as e:
            print(f"Error reading text file: {e}")

        current_id = 0
        # Add special characters
        for token in self.special_tokens:
            if token not in self.char_to_idx:
                self.char_to_idx[token] = current_id
                self.idx_to_char[current_id] = token
                current_id += 1
        # Add other characters
        sorted_ordinary_chars = sorted(list(all_chars - set(self.special_tokens)))
        for char in sorted_ordinary_chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = current_id
                self.idx_to_char[current_id] = char
                current_id += 1
        # Vocabulary size
        self.vocab_size = len(self.char_to_idx)

        self.pad_token_id = self.char_to_idx.get('<PAD>')
        self.unk_token_id = self.char_to_idx.get('<UNK>')
        self.bos_token_id = self.char_to_idx.get('<BOS>')
        self.eos_token_id = self.char_to_idx.get('<EOD')

    def encode(self, text):
        return [self.char_to_idx.get(char, self.unk_token_id) for char in text]

    def decode(self, ids):
        return "".join([self.idx_to_char.get(idx,'') for idx in ids])

    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_token_id(self):
        return self.pad_token_id

    def get_unk_token_id(self):
        return self.unk_token_id

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_eos_token_id(self):
        return self.eos_token_id
