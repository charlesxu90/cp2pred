from abc import ABC
import logging
import torch
from .bpe import BPEEncoder

logger = logging.getLogger(__name__)


class Tokenizer(ABC):
    PAD = '<PAD>'
    BEGIN = '<SOS>'
    END = '<EOS>'
    MASK = '<MASK>'

    def __init__(self, max_len=100) -> None:
        self.char_idx = {}
        self.idx_char = {}
        self.max_len = max_len
        pass

    def preprocess_str(self, seq):
        return seq

    def postprocess_str(self, seq):
        return seq

    def tokenize(self,  seqs):
        # logger.debug(f'max_len: {self.max_len}')
        batch_size = len(seqs)
        idx_matrix = torch.zeros((batch_size, self.max_len))
        for i, seq in enumerate(seqs):
            enc_seq = self.BEGIN + self.preprocess_str(seq) + self.END
            for j in range(self.max_len):
                if j >= len(enc_seq):
                    break
                idx_matrix[i, j] = self.char_idx[enc_seq[j]]

        return idx_matrix.to(torch.int64)

    def detokenize(self, token_array):
        seqs_strings = []

        for row in token_array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            seq = ''.join(predicted_chars)
            seq = self.postprocess_str(seq)
            seqs_strings.append(seq)

        return seqs_strings
    
    def get_vocab_size(self) -> int:
        return len(self.idx_char)
    
    @property
    def begin_token_id(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_token_id(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_token_id(self) -> int:
        return self.char_idx[self.PAD]

    @property
    def mask_token_id(self) -> int:
        return self.char_idx[self.MASK]


class SmilesTokenizer(Tokenizer):
    """
    A fixed dictionary for SMILES sequences.
    """
    def __init__(self, max_len=189) -> None:
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                         '.': 30, '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                         '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                         'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                         'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
                         "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46, self.MASK: 47,
                         }

        self.idx_char = {v: k for k, v in self.char_idx.items()}
        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.max_len = max_len

    def allowed(self, smiles) -> bool:
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                logger.info('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True

    def preprocess_str(self, smiles: str) -> str:
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def postprocess_str(self, smiles):
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(token, symbol)
        return temp_smiles


class AATokenizer(Tokenizer):
    """
    A fixed dictionary for protein sequences.
    """
    def __init__(self, max_len=40) -> None:
        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, 'A': 3, 'R': 4, 'N': 5, 'D': 6, 'C': 7, 'E': 8,
                         'Q': 9, 'G': 10, 'H': 11, 'I': 12, 'L': 13, 'K': 14, 'M': 15, 'F': 16, 'P': 17, 'S': 18,
                         'T': 19, 'W': 20, 'Y': 21, 'V': 22, 'X': 23, self.MASK: 24}  # X for unknown AA
        self.idx_char = {v: k for k, v in self.char_idx.items()}
        self.max_len = max_len


class HELMTokenizer(Tokenizer):
    """
    A fixed dictionary for HELM sequences.
    """

    def __init__(self, max_len=100) -> None:
        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, 
                         'A': 3, 'R': 4, 'N': 5, 'D': 6, 'C': 7, 'E': 8,
                         'Q': 9, 'G': 10, 'H': 11, 'I': 12, 'L': 13, 'K': 14, 'M': 15, 'F': 16, 'P': 17, 'S': 18,
                         'T': 19, 'W': 20, 'Y': 21, 'V': 22, 'X': 23, # Natural amino acids
                         '$': 24,  '(': 25,  ')': 26,  ',': 27,  '-': 28,  '.': 29,  ':': 30,  '[': 31,  ']': 32,  '{': 33,  '|': 34,  '}': 35,  # Common symbols in HELM
                         '0': 36,  '1': 37,  '2': 38,  '3': 39,  '4': 40,  '5': 41,  '6': 42,  '7': 43,  '8': 44,  '9': 45,  
                         '>': 46,  'B': 47,  'O': 48,  '_': 49,  'a': 50,  'b': 51,  'c': 52,  'd': 53,  'e': 54,  'f': 55,  'g': 56,  'h': 57,  'i': 58,  'l': 59,  'm': 60,  'n': 61,  'o': 62,  'p': 63,  'r': 64,  's': 65,  't': 66,  'u': 67,  'v': 68,  'x': 69,  'y': 70,  'z': 71,
                         self.MASK: 72}
        self.idx_char = {v: k for k, v in self.char_idx.items()}
        self.max_len = max_len


class BPETokenizer(Tokenizer):
    """
    A simple BPE tokenizer for different sequences.
    """
    PAD = '<PAD>'
    BEGIN = '<SOS>'
    END = '<EOS>'
    MASK = '<MASK>'
    UNK = '<UNK>'
    CLS = '<CLS>'

    def __init__(self, bpe_path='results/helm_bpe', max_len=100):
        super().__init__()
        self.bpe_encoder = BPEEncoder.load(bpe_path)
        self.max_len = max_len

    def tokenize(self, seqs):
        batch_size = len(seqs)
        idx_matrix = torch.zeros((batch_size, self.max_len))
        for i, seq in enumerate(seqs):
            enc_seq = self.BEGIN + seq + self.END
            enc_idx = self.bpe_encoder.encode(enc_seq)
            if len(enc_idx) > self.max_len:
                idx_matrix[i, :self.max_len] = torch.tensor(enc_idx[:self.max_len])
            else:
                idx_matrix[i, :len(enc_idx)] = torch.tensor(enc_idx)
        return idx_matrix.to(torch.int64)

    def detokenize(self, token_array):
        seqs_strings = []

        for row in token_array:
            seq = self.bpe_encoder.decode(row.tolist())
            seq = seq.replace(self.BEGIN, '')
            
            if self.END in seq:  # remove the chars after the first END
                seq = seq[:seq.index(self.END)]
            seqs_strings.append(seq)

        return seqs_strings
    
    def get_vocab_size(self) -> int:
        return self.bpe_encoder.get_vocab_size()
    
    @property
    def begin_token_id(self) -> int:
        return self.bpe_encoder.encoder[self.BEGIN]

    @property
    def end_token_id(self) -> int:
        return self.bpe_encoder.encoder[self.END]

    @property
    def pad_token_id(self) -> int:
        return self.bpe_encoder.encoder[self.PAD]

    @property
    def mask_token_id(self) -> int:
        return self.bpe_encoder.encoder[self.MASK]
